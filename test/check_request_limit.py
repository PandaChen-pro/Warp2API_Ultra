#!/usr/bin/env python3
"""
Warp Account Request Limit Checker
获取Warp账户的请求额度信息
"""

import asyncio
import json
import sqlite3
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import httpx
import platform


class WarpRequestLimitChecker:
    """Warp账户请求额度检查器"""

    def __init__(self, db_path: str = "../warp_accounts.db"):
        """
        初始化检查器

        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path
        self.async_client = httpx.AsyncClient(timeout=30.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.async_client.aclose()

    def get_account_from_db(self, email: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        从数据库获取账户信息

        Args:
            email: 账户邮箱，如果为None则获取第一个active账户

        Returns:
            账户信息字典或None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if email:
                cursor.execute("""
                               SELECT *
                               FROM accounts
                               WHERE email = ?
                                 AND status = 'active'
                               """, (email,))
            else:
                cursor.execute("""
                               SELECT *
                               FROM accounts
                               WHERE status = 'active'
                               ORDER BY last_used ASC, id ASC LIMIT 1
                               """)

            row = cursor.fetchone()
            conn.close()

            if row:
                return dict(row)
            return None

        except Exception as e:
            print(f"❌ 数据库查询错误: {e}")
            return None

    async def get_request_limit(self, id_token: str) -> Dict[str, Any]:
        """
        获取账户请求额度

        Args:
            id_token: Firebase ID Token

        Returns:
            包含额度信息的字典
        """
        if not id_token:
            return {"success": False, "error": "缺少Firebase ID Token"}

        try:
            url = "https://app.warp.dev/graphql/v2"

            # GraphQL查询
            query = """query GetRequestLimitInfo($requestContext: RequestContext!) {
  user(requestContext: $requestContext) {
    __typename
    ... on UserOutput {
      user {
        requestLimitInfo {
          isUnlimited
          nextRefreshTime
          requestLimit
          requestsUsedSinceLastRefresh
          requestLimitRefreshDuration
          isUnlimitedAutosuggestions
          acceptedAutosuggestionsLimit
          acceptedAutosuggestionsSinceLastRefresh
          isUnlimitedVoice
          voiceRequestLimit
          voiceRequestsUsedSinceLastRefresh
          voiceTokenLimit
          voiceTokensUsedSinceLastRefresh
          isUnlimitedCodebaseIndices
          maxCodebaseIndices
          maxFilesPerRepo
          embeddingGenerationBatchSize
        }
      }
    }
    ... on UserFacingError {
      error {
        __typename
        ... on SharedObjectsLimitExceeded {
          limit
          objectType
          message
        }
        ... on PersonalObjectsLimitExceeded {
          limit
          objectType
          message
        }
        ... on AccountDelinquencyError {
          message
        }
        ... on GenericStringObjectUniqueKeyConflict {
          message
        }
      }
      responseContext {
        serverVersion
      }
    }
  }
}
"""

            # 系统信息
            os_category = "Web"
            os_name = "Windows"
            os_version = "NT 10.0"
            app_version = "v0.2025.10.01.08.12.stable_02"

            data = {
                "operationName": "GetRequestLimitInfo",
                "variables": {
                    "requestContext": {
                        "clientContext": {
                            "version": app_version
                        },
                        "osContext": {
                            "category": os_category,
                            "linuxKernelVersion": None,
                            "name": os_name,
                            "version": os_version
                        }
                    }
                },
                "query": query
            }

            headers = {
                "Content-Type": "application/json",
                "authorization": f"Bearer {id_token}",
                "x-warp-client-id": "warp-app",
                "x-warp-client-version": app_version,
                "x-warp-os-category": os_category,
                "x-warp-os-name": os_name,
                "x-warp-os-version": os_version,
            }

            print("📊 调用GetRequestLimitInfo接口...")

            response = await self.async_client.post(
                url,
                params={"op": "GetRequestLimitInfo"},
                json=data,
                headers=headers,
            )

            if response.status_code == 200:
                result = response.json()

                # 检查错误
                if "errors" in result:
                    error_msg = result["errors"][0].get("message", "Unknown error")
                    print(f"❌ GraphQL错误: {error_msg}")
                    return {"success": False, "error": error_msg}

                # 解析响应
                data_result = result.get("data", {})
                user_data = data_result.get("user", {})

                if user_data.get("__typename") == "UserOutput":
                    user_info = user_data.get("user", {})
                    request_limit_info = user_info.get("requestLimitInfo", {})

                    # 获取额度信息
                    request_limit = request_limit_info.get("requestLimit", 0)
                    requests_used = request_limit_info.get("requestsUsedSinceLastRefresh", 0)
                    is_unlimited = request_limit_info.get("isUnlimited", False)
                    next_refresh_time = request_limit_info.get("nextRefreshTime", "N/A")
                    refresh_duration = request_limit_info.get("requestLimitRefreshDuration", "WEEKLY")

                    # 计算剩余额度
                    requests_remaining = request_limit - requests_used

                    # 判断额度类型
                    if is_unlimited:
                        quota_type = "🚀 无限额度"
                    elif request_limit >= 2500:
                        quota_type = "🎉 高额度"
                    else:
                        quota_type = "📋 普通额度"

                    print(f"\n✅ 账户额度信息:")
                    print(f"   {quota_type}: {request_limit}")
                    print(f"   📊 已使用: {requests_used}/{request_limit}")
                    print(f"   💎 剩余: {requests_remaining}")
                    print(f"   🔄 刷新周期: {refresh_duration}")
                    print(f"   ⏰ 下次刷新: {next_refresh_time}")

                    # 额外限制信息
                    if request_limit_info.get("isUnlimitedAutosuggestions"):
                        print(f"   ✨ 自动建议: 无限制")
                    if request_limit_info.get("maxCodebaseIndices"):
                        print(f"   📚 最大代码库索引: {request_limit_info.get('maxCodebaseIndices')}")

                    return {
                        "success": True,
                        "requestLimit": request_limit,
                        "requestsUsed": requests_used,
                        "requestsRemaining": requests_remaining,
                        "isUnlimited": is_unlimited,
                        "nextRefreshTime": next_refresh_time,
                        "refreshDuration": refresh_duration,
                        "quotaType": "unlimited" if is_unlimited else ("high" if request_limit >= 2500 else "normal"),
                        "autosuggestions": {
                            "isUnlimited": request_limit_info.get("isUnlimitedAutosuggestions", False),
                            "limit": request_limit_info.get("acceptedAutosuggestionsLimit", 0),
                            "used": request_limit_info.get("acceptedAutosuggestionsSinceLastRefresh", 0)
                        },
                        "voice": {
                            "isUnlimited": request_limit_info.get("isUnlimitedVoice", False),
                            "requestLimit": request_limit_info.get("voiceRequestLimit", 0),
                            "requestsUsed": request_limit_info.get("voiceRequestsUsedSinceLastRefresh", 0),
                            "tokenLimit": request_limit_info.get("voiceTokenLimit", 0),
                            "tokensUsed": request_limit_info.get("voiceTokensUsedSinceLastRefresh", 0)
                        },
                        "codebase": {
                            "isUnlimited": request_limit_info.get("isUnlimitedCodebaseIndices", False),
                            "maxIndices": request_limit_info.get("maxCodebaseIndices", 0),
                            "maxFilesPerRepo": request_limit_info.get("maxFilesPerRepo", 0)
                        }
                    }

                elif user_data.get("__typename") == "UserFacingError":
                    error = user_data.get("error", {}).get("message", "Unknown error")
                    print(f"❌ 获取额度失败: {error}")
                    return {"success": False, "error": error}
                else:
                    print(f"❌ 响应中没有找到用户信息")
                    return {"success": False, "error": "未找到用户信息"}

            else:
                error_text = response.text[:500]
                print(f"❌ HTTP错误 {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}: {error_text}"}

        except Exception as e:
            print(f"❌ 获取额度错误: {e}")
            return {"success": False, "error": str(e)}

    def update_account_usage(self, email: str) -> bool:
        """
        更新账户使用信息

        Args:
            email: 账户邮箱

        Returns:
            是否更新成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                           UPDATE accounts
                           SET last_used  = CURRENT_TIMESTAMP,
                               use_count  = use_count + 1,
                               updated_at = CURRENT_TIMESTAMP
                           WHERE email = ?
                           """, (email,))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"❌ 更新账户使用信息失败: {e}")
            return False


async def check_single_account(email: Optional[str] = None):
    """
    检查单个账户的请求额度

    Args:
        email: 账户邮箱，如果为None则检查第一个active账户
    """
    async with WarpRequestLimitChecker() as checker:
        # 获取账户信息
        account = checker.get_account_from_db(email)

        if not account:
            print(f"❌ 未找到账户: {email if email else '没有active账户'}")
            return

        print(f"\n🔍 检查账户: {account['email']}")
        print(f"   📅 创建时间: {account['created_at']}")
        print(f"   🔢 使用次数: {account['use_count']}")
        print(f"   ⏱️ 上次使用: {account['last_used']}")

        # 获取请求额度
        result = await checker.get_request_limit(account['id_token'])

        if result['success']:
            # 更新账户使用信息
            checker.update_account_usage(account['email'])

            # 保存结果到文件
            output_file = f"request_limit_{account['email'].split('@')[0]}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 结果已保存到: {output_file}")

        return result


async def check_all_accounts():
    """检查所有active账户的请求额度"""
    async with WarpRequestLimitChecker() as checker:
        # 获取所有active账户
        conn = sqlite3.connect(checker.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT email, id_token
                       FROM accounts
                       WHERE status = 'active'
                       ORDER BY id
                       """)

        accounts = cursor.fetchall()
        conn.close()

        if not accounts:
            print("❌ 没有找到active账户")
            return

        print(f"📋 找到 {len(accounts)} 个active账户")

        results = []
        for idx, account in enumerate(accounts, 1):
            print(f"\n========== [{idx}/{len(accounts)}] ==========")
            print(f"🔍 检查账户: {account['email']}")

            result = await checker.get_request_limit(account['id_token'])
            result['email'] = account['email']
            results.append(result)

            # 更新使用信息
            if result['success']:
                checker.update_account_usage(account['email'])

            # 避免请求过快
            if idx < len(accounts):
                await asyncio.sleep(1)

        # 保存所有结果
        output_file = f"all_accounts_limit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 所有结果已保存到: {output_file}")

        # 统计信息
        print("\n📊 统计摘要:")
        success_count = sum(1 for r in results if r['success'])
        unlimited_count = sum(1 for r in results if r.get('success') and r.get('isUnlimited'))
        high_quota_count = sum(1 for r in results if r.get('success') and r.get('quotaType') == 'high')
        normal_quota_count = sum(1 for r in results if r.get('success') and r.get('quotaType') == 'normal')

        print(f"   ✅ 成功检查: {success_count}/{len(accounts)}")
        print(f"   🚀 无限额度: {unlimited_count}")
        print(f"   🎉 高额度账户: {high_quota_count}")
        print(f"   📋 普通额度账户: {normal_quota_count}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Warp账户请求额度检查器")
    parser.add_argument("--email", help="指定要检查的账户邮箱")
    parser.add_argument("--all", action="store_true", help="检查所有active账户")
    parser.add_argument("--db", default="warp_accounts.db", help="数据库路径")
    parser.add_argument("--test", action="store_true", help="使用测试数据")

    args = parser.parse_args()

    if args.test:
        # 使用提供的测试数据
        test_id_token = ""

        async def test_with_token():
            async with WarpRequestLimitChecker() as checker:
                print("🧪 测试模式 - 使用提供的ID Token")
                result = await checker.get_request_limit(test_id_token)

                if result['success']:
                    print("\n✅ 测试成功!")
                    with open("test_result.json", 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print("💾 结果已保存到: test_result.json")
                else:
                    print(f"\n❌ 测试失败: {result.get('error')}")

        asyncio.run(test_with_token())

    elif args.all:
        asyncio.run(check_all_accounts())
    else:
        asyncio.run(check_single_account(args.email))


if __name__ == "__main__":
    main()
