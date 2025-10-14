#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warp API客户端模块

处理与Warp API的通信，包括protobuf数据发送和SSE响应解析。
"""
import asyncio
import os
from typing import Any, Dict, LiteralString

import httpx

from ..config.settings import WARP_URL as CONFIG_WARP_URL
from ..core.logging import logger
from ..core.pool_auth import acquire_pool_session_with_info, release_pool_session
from ..core.protobuf_utils import protobuf_to_dict

# 可配置的重试参数
MAX_QUOTA_RETRIES = 5
RETRY_DELAY_SECONDS = 0.2


def _get(d: Dict[str, Any], *names: str) -> Any:
    """Return the first matching key value (camelCase/snake_case tolerant)."""
    for name in names:
        if name in d:
            return d[name]
    return None


def _get_event_type(event_data: dict) -> str:
    """Determine the type of SSE event for logging"""
    if "init" in event_data:
        return "INITIALIZATION"
    client_actions = _get(event_data, "client_actions", "clientActions")
    if isinstance(client_actions, dict):
        actions = _get(client_actions, "actions", "Actions") or []
        if not actions:
            return "CLIENT_ACTIONS_EMPTY"

        action_types = []
        for action in actions:
            if _get(action, "create_task", "createTask") is not None:
                action_types.append("CREATE_TASK")
            elif _get(action, "append_to_message_content", "appendToMessageContent") is not None:
                action_types.append("APPEND_CONTENT")
            elif _get(action, "add_messages_to_task", "addMessagesToTask") is not None:
                action_types.append("ADD_MESSAGE")
            elif _get(action, "update_task_message", "updateTaskMessage") is not None:
                action_types.append("UPDATE_MESSAGE")
            elif _get(action, "tool_call", "toolCall") is not None:
                action_types.append("TOOL_CALL")
            elif _get(action, "tool_response", "toolResponse") is not None:
                action_types.append("TOOL_RESPONSE")
            elif _get(action, "begin_transaction", "beginTransaction") is not None:
                action_types.append("BEGIN_TRANSACTION")
            elif _get(action, "rollback_transaction", "rollbackTransaction") is not None:
                action_types.append("ROLLBACK_TRANSACTION")
            else:
                action_types.append("UNKNOWN_ACTION")

        return f"CLIENT_ACTIONS({', '.join(action_types)})"
    elif "finished" in event_data:
        return "FINISHED"
    else:
        return "UNKNOWN_EVENT"


def _extract_text_from_message(message: Dict[str, Any]) -> str:
    """
    增强版文本提取函数，检查消息对象的多个可能位置以提取文本内容
    """
    if not isinstance(message, dict):
        return ""

    # 1. 检查 agent_output.text (最常见)
    agent_output = _get(message, "agent_output", "agentOutput")
    if isinstance(agent_output, dict):
        text = agent_output.get("text", "")
        if text:
            return text

    # 2. 检查 content 字段的多种结构
    content = _get(message, "content", "Content")
    if isinstance(content, dict):
        # 2.1 直接的 text 字段
        if "text" in content and isinstance(content["text"], str):
            return content["text"]

        # 2.2 parts 数组结构
        parts = content.get("parts", content.get("Parts", []))
        if isinstance(parts, list) and parts:
            text_parts = []
            for part in parts:
                if isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            if text_parts:
                return "".join(text_parts)

    # 3. 检查顶层的 text 字段
    if "text" in message and isinstance(message["text"], str):
        return message["text"]

    # 4. 检查 user_query 字段（用于用户消息）
    user_query = _get(message, "user_query", "userQuery")
    if isinstance(user_query, dict):
        text = user_query.get("text", "")
        if text:
            return text
    elif isinstance(user_query, str):
        return user_query

    return ""


async def send_protobuf_to_warp_api(
        protobuf_bytes: bytes, show_all_events: bool = True
) -> None | tuple[str, None, None] | tuple[LiteralString, Any | None, Any | None] | tuple[str, Any | None, Any | None]:
    """发送protobuf数据到Warp API并获取响应，支持动态代理和SSL错误重试"""
    # 导入代理管理器
    from ..core.proxy_manager import AsyncProxyManager
    proxy_manager = AsyncProxyManager()

    max_proxy_retries = 3  # 每次配额重试使用新代理

    # 用于跟踪当前会话信息
    current_session = None

    try:
        logger.info(f"发送 {len(protobuf_bytes)} 字节到Warp API")
        logger.info(f"数据包前32字节 (hex): {protobuf_bytes[:32].hex()}")

        warp_url = CONFIG_WARP_URL
        logger.info(f"发送请求到: {warp_url}")

        conversation_id = None
        task_id = None
        complete_response = []
        all_events = []
        event_count = 0

        verify_opt = False  # 使用代理时关闭SSL验证
        insecure_env = os.getenv("WARP_INSECURE_TLS", "").lower()
        if insecure_env in ("1", "true", "yes"):
            verify_opt = False
            logger.warning("TLS verification disabled via WARP_INSECURE_TLS for Warp API client")

        # 主重试循环（用于配额用尽等可恢复错误）
        for attempt in range(MAX_QUOTA_RETRIES):
            # 释放之前的会话（如果有）
            if current_session:
                await release_pool_session(current_session.get("session_id"))
                current_session = None

            # 获取新的会话
            current_session = await acquire_pool_session_with_info()
            if not current_session or not current_session.get("access_token"):
                logger.error("无法获取有效的认证会话，请求中止。")
                return f"❌ Error: Could not acquire auth session", None, None

            jwt = current_session["access_token"]
            account_email = current_session.get("account", {}).get("email", "unknown")
            logger.info(f"使用账号 {account_email} 进行请求 (attempt {attempt + 1}/{MAX_QUOTA_RETRIES})")

            # 代理重试循环
            for proxy_attempt in range(max_proxy_retries):
                try:
                    # 获取新的代理
                    proxy_str = await proxy_manager.get_proxy()
                    proxy_config = None

                    if proxy_str:
                        proxy_config = proxy_manager.format_proxy_for_httpx(proxy_str)
                    else:
                        logger.warning("无法获取代理，使用直连")

                    # 创建带代理的客户端
                    client_config = {
                        "http2": True,
                        "timeout": httpx.Timeout(60.0),
                        "verify": verify_opt,
                        "trust_env": True
                    }

                    # 如果有代理配置，添加代理参数
                    if proxy_config:
                        client_config["proxies"] = proxy_config

                    async with httpx.AsyncClient(**client_config) as client:
                        headers = {
                            "accept": "text/event-stream",
                            "content-type": "application/x-protobuf",
                            "x-warp-client-version": "v0.2025.08.06.08.12.stable_02",
                            "x-warp-os-category": "Windows",
                            "x-warp-os-name": "Windows",
                            "x-warp-os-version": "11 (26100)",
                            "authorization": f"Bearer {jwt}",
                            "content-length": str(len(protobuf_bytes)),
                        }

                        async with client.stream("POST", warp_url, headers=headers, content=protobuf_bytes) as response:
                            # 如果请求成功，处理响应
                            if response.status_code == 200:
                                logger.info(f"✅ 收到HTTP {response.status_code}响应")
                                logger.info("开始处理SSE事件流...")

                                import re as _re
                                def _parse_payload_bytes(data_str: str):
                                    s = _re.sub(r"\\s+", "", data_str or "")
                                    if not s: return None
                                    if _re.fullmatch(r"[0-9a-fA-F]+", s or ""):
                                        try:
                                            return bytes.fromhex(s)
                                        except Exception:
                                            pass
                                    pad = "=" * ((4 - (len(s) % 4)) % 4)
                                    try:
                                        import base64 as _b64
                                        return _b64.urlsafe_b64decode(s + pad)
                                    except Exception:
                                        try:
                                            return _b64.b64decode(s + pad)
                                        except Exception:
                                            return None

                                current_data = ""

                                async for line in response.aiter_lines():
                                    if line.startswith("data:"):
                                        payload = line[5:].strip()
                                        if not payload: continue
                                        if payload == "[DONE]":
                                            logger.info("收到[DONE]标记，结束处理")
                                            break
                                        current_data += payload
                                        continue

                                    if (line.strip() == "") and current_data:
                                        raw_bytes = _parse_payload_bytes(current_data)
                                        current_data = ""
                                        if raw_bytes is None:
                                            logger.debug("跳过无法解析的SSE数据块（非hex/base64或不完整）")
                                            continue
                                        try:
                                            event_data = protobuf_to_dict(raw_bytes,
                                                                          "warp.multi_agent.v1.ResponseEvent")
                                        except Exception as parse_error:
                                            logger.debug(f"解析事件失败，跳过: {str(parse_error)[:100]}")
                                            continue
                                        event_count += 1

                                        def _get(d: Dict[str, Any], *names: str) -> Any:
                                            for n in names:
                                                if isinstance(d, dict) and n in d:
                                                    return d[n]
                                            return None

                                        event_type = _get_event_type(event_data)
                                        if show_all_events:
                                            all_events.append(
                                                {"event_number": event_count, "event_type": event_type,
                                                 "raw_data": event_data})
                                        logger.info(f"🔄 Event #{event_count}: {event_type}")
                                        if show_all_events:
                                            logger.info(f"   📋 Event data: {str(event_data)}")

                                        if "init" in event_data:
                                            init_data = event_data["init"]
                                            conversation_id = init_data.get("conversation_id", conversation_id)
                                            task_id = init_data.get("task_id", task_id)
                                            logger.info(f"会话初始化: {conversation_id}")

                                        client_actions = _get(event_data, "client_actions", "clientActions")
                                        if isinstance(client_actions, dict):
                                            actions = _get(client_actions, "actions", "Actions") or []
                                            for i, action in enumerate(actions):
                                                logger.info(f"   🎯 Action #{i + 1}: {list(action.keys())}")

                                                # 处理 update_task_message（新增）
                                                update_msg_data = _get(action, "update_task_message",
                                                                       "updateTaskMessage")
                                                if isinstance(update_msg_data, dict):
                                                    message = update_msg_data.get("message", {})
                                                    text_content = _extract_text_from_message(message)
                                                    if text_content:
                                                        complete_response.append(text_content)
                                                        logger.info(
                                                            f"   📝 Text from UPDATE_MESSAGE: {text_content}")

                                                # 处理 append_to_message_content
                                                append_data = _get(action, "append_to_message_content",
                                                                   "appendToMessageContent")
                                                if isinstance(append_data, dict):
                                                    message = append_data.get("message", {})
                                                    agent_output = _get(message, "agent_output", "agentOutput") or {}
                                                    text_content = agent_output.get("text", "")
                                                    if text_content:
                                                        complete_response.append(text_content)
                                                        logger.info(f"   📝 Text Fragment: {text_content}")

                                                # 处理 add_messages_to_task
                                                messages_data = _get(action, "add_messages_to_task",
                                                                     "addMessagesToTask")
                                                if isinstance(messages_data, dict):
                                                    messages = messages_data.get("messages", [])
                                                    task_id = messages_data.get("task_id",
                                                                                messages_data.get("taskId", task_id))
                                                    for j, message in enumerate(messages):
                                                        logger.info(f"   📨 Message #{j + 1}: {list(message.keys())}")
                                                        text_content = _extract_text_from_message(message)
                                                        if text_content:
                                                            complete_response.append(text_content)
                                                            logger.info(
                                                                f"   📝 Complete Message: {text_content}")

                                full_response = "".join(complete_response)
                                logger.info("=" * 60)
                                logger.info("📊 SSE STREAM SUMMARY")
                                logger.info("=" * 60)
                                logger.info(f"📈 Total Events Processed: {event_count}")
                                logger.info(f"🆔 Conversation ID: {conversation_id}")
                                logger.info(f"🆔 Task ID: {task_id}")
                                logger.info(f"📝 Response Length: {len(full_response)} characters")
                                logger.info("=" * 60)

                                # 成功完成，释放会话并返回结果
                                await release_pool_session(current_session.get("session_id"))
                                current_session = None

                                if full_response:
                                    logger.info(f"✅ Stream processing completed successfully")
                                    return full_response, conversation_id, task_id
                                else:
                                    logger.warning("⚠️ No text content received in response")
                                    return "Warning: No response content received", conversation_id, task_id

                            # --- 处理错误响应 ---
                            error_text = await response.aread()
                            error_content = error_text.decode('utf-8') if error_text else "No error content"

                            # 检查是否是账号被封禁错误 (403)
                            is_blocked_error = (
                                    response.status_code == 403 and (
                                    ("Your account has been blocked" in error_content) or
                                    ("blocked from using AI features" in error_content)
                            )
                            )

                            if is_blocked_error:
                                logger.error(f"❌ 账号 {account_email} 已被封禁 (HTTP 403)")
                                # 释放并标记为blocked
                                if current_session:
                                    # 通知pool service标记账号
                                    try:
                                        async with httpx.AsyncClient(timeout=5.0) as notify_client:
                                            await notify_client.post(
                                                "http://localhost:8019/api/accounts/mark_blocked",
                                                json={"email": account_email}
                                            )
                                    except:
                                        pass

                                    await release_pool_session(current_session.get("session_id"))
                                    current_session = None

                                # 如果还有重试次数，获取新账号
                                if attempt < (MAX_QUOTA_RETRIES - 1):
                                    logger.warning(
                                        f"账号被封，将获取新账号重试 (第 {attempt + 2}/{MAX_QUOTA_RETRIES} 次)...")
                                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                                    break  # 跳出代理循环，进入下一个attempt获取新账号
                                else:
                                    return f"❌ Account blocked after {MAX_QUOTA_RETRIES} attempts", None, None

                            # 检查是否是配额用尽错误
                            is_quota_error = ("No remaining quota" in error_content) or (
                                    "No AI requests remaining" in error_content)

                            if response.status_code == 429 and is_quota_error:
                                if attempt < (MAX_QUOTA_RETRIES - 1):
                                    logger.warning(
                                        f"Warp API 返回 429 (配额用尽)。将在 {RETRY_DELAY_SECONDS} 秒后强制获取新账号并重试 (第 {attempt + 2}/{MAX_QUOTA_RETRIES} 次)...")
                                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                                    # 跳出代理循环，进入下一个attempt获取新账号
                                    break
                                else:
                                    # 所有账号都用尽了
                                    await release_pool_session(current_session.get("session_id"))
                                    current_session = None
                                    return f"❌ API Error (HTTP {response.status_code}) after {MAX_QUOTA_RETRIES} attempts: {error_content}", None, None

                            # 其他HTTP错误，尝试换代理
                            logger.error(
                                f"HTTP错误 {response.status_code}，尝试换代理 (proxy attempt {proxy_attempt + 1}/{max_proxy_retries})")
                            if proxy_attempt < max_proxy_retries - 1:
                                await asyncio.sleep(0.5)
                                continue  # 继续下一个proxy_attempt

                            # 所有代理都失败，如果还有账号重试次数，换账号
                            if attempt < (MAX_QUOTA_RETRIES - 1):
                                logger.warning(f"当前账号的所有代理都失败，将换新账号重试")
                                break  # 跳出代理循环

                            # 真正失败了
                            await release_pool_session(current_session.get("session_id"))
                            current_session = None
                            return f"❌ API Error (HTTP {response.status_code}): {error_content}", None, None

                except (httpx.ConnectError, httpx.ProxyError, httpx.RemoteProtocolError) as ssl_error:
                    logger.warning(f"SSL/代理错误 (proxy attempt {proxy_attempt + 1}/{max_proxy_retries}): {ssl_error}")
                    if proxy_attempt < max_proxy_retries - 1:
                        await asyncio.sleep(0.5)
                        continue
                    # 所有代理都失败，进入下一个attempt
                    break

                except httpx.ReadTimeout:
                    logger.warning(f"请求超时，尝试换代理 (proxy attempt {proxy_attempt + 1}/{max_proxy_retries})")
                    if proxy_attempt < max_proxy_retries - 1:
                        await asyncio.sleep(0.5)
                        continue
                    break

                except Exception as e:
                    logger.error(f"未知错误: {e}")
                    if proxy_attempt < max_proxy_retries - 1:
                        await asyncio.sleep(0.5)
                        continue
                    raise

    except Exception as e:
        import traceback
        logger.error("=" * 60)
        logger.error("WARP API CLIENT EXCEPTION")
        logger.error("=" * 60)
        logger.error(f"Exception Type: {type(e).__name__}")
        logger.error(f"Exception Message: {str(e)}")
        logger.error(f"Request Size: {len(protobuf_bytes) if 'protobuf_bytes' in locals() else 'Unknown'}")
        logger.error("Python Traceback:")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
        raise
    finally:
        # 确保会话被释放
        if current_session:
            await release_pool_session(current_session.get("session_id"))


async def send_protobuf_to_warp_api_parsed(protobuf_bytes: bytes) -> None | tuple[str, None, None, list[Any]] | tuple[LiteralString, Any | None, Any | None, list[Any]]:
    """发送protobuf数据到Warp API并获取解析后的SSE事件数据，支持动态代理和SSL错误重试"""
    # 导入代理管理器
    from ..core.proxy_manager import AsyncProxyManager
    proxy_manager = AsyncProxyManager()

    max_proxy_retries = 3  # 每次配额重试使用新代理

    # 用于跟踪当前会话信息
    current_session = None

    try:
        logger.info(f"发送 {len(protobuf_bytes)} 字节到Warp API (解析模式)")
        logger.info(f"数据包前32字节 (hex): {protobuf_bytes[:32].hex()}")

        warp_url = CONFIG_WARP_URL
        logger.info(f"发送请求到: {warp_url}")

        conversation_id = None
        task_id = None
        complete_response = []
        parsed_events = []
        event_count = 0

        verify_opt = False  # 使用代理时关闭SSL验证
        insecure_env = os.getenv("WARP_INSECURE_TLS", "").lower()
        if insecure_env in ("1", "true", "yes"):
            verify_opt = False
            logger.warning("TLS verification disabled via WARP_INSECURE_TLS for Warp API client")

        # 重试循环
        for attempt in range(MAX_QUOTA_RETRIES):
            # 释放之前的会话（如果有）
            if current_session:
                await release_pool_session(current_session.get("session_id"))
                current_session = None

            # 获取新的会话
            current_session = await acquire_pool_session_with_info()
            if not current_session or not current_session.get("access_token"):
                logger.error("无法获取有效的认证会话，请求中止（解析模式）。")
                return f"❌ Error: Could not acquire auth session", None, None, []

            jwt = current_session["access_token"]
            account_email = current_session.get("account", {}).get("email", "unknown")
            logger.info(f"使用账号 {account_email} 进行请求 (解析模式, attempt {attempt + 1}/{MAX_QUOTA_RETRIES})")

            for proxy_attempt in range(max_proxy_retries):
                try:
                    # 获取新的代理
                    proxy_str = await proxy_manager.get_proxy()
                    proxy_config = None

                    if proxy_str:
                        proxy_config = proxy_manager.format_proxy_for_httpx(proxy_str)
                    else:
                        logger.warning("无法获取代理，使用直连(解析模式)")

                    # 创建带代理的客户端
                    client_config = {
                        "http2": True,
                        "timeout": httpx.Timeout(60.0),
                        "verify": verify_opt,
                        "trust_env": True
                    }

                    # 如果有代理配置，添加代理参数
                    if proxy_config:
                        client_config["proxy"] = proxy_config

                    async with httpx.AsyncClient(**client_config) as client:
                        headers = {
                            "accept": "text/event-stream",
                            "content-type": "application/x-protobuf",
                            "x-warp-client-version": "v0.2025.08.06.08.12.stable_02",
                            "x-warp-os-category": "Windows",
                            "x-warp-os-name": "Windows",
                            "x-warp-os-version": "11 (26100)",
                            "authorization": f"Bearer {jwt}",
                            "content-length": str(len(protobuf_bytes)),
                        }

                        async with client.stream("POST", warp_url, headers=headers, content=protobuf_bytes) as response:
                            # 如果请求成功，在这里处理响应
                            if response.status_code == 200:
                                logger.info(f"✅ 收到HTTP {response.status_code}响应 (解析模式)")
                                logger.info("开始处理SSE事件流...")

                                # 处理响应流
                                import re as _re2
                                def _parse_payload_bytes2(data_str: str):
                                    s = _re2.sub(r"\\s+", "", data_str or "")
                                    if not s: return None
                                    if _re2.fullmatch(r"[0-9a-fA-F]+", s or ""):
                                        try:
                                            return bytes.fromhex(s)
                                        except Exception:
                                            pass
                                    pad = "=" * ((4 - (len(s) % 4)) % 4)
                                    try:
                                        import base64 as _b642
                                        return _b642.urlsafe_b64decode(s + pad)
                                    except Exception:
                                        try:
                                            return _b642.b64decode(s + pad)
                                        except Exception:
                                            return None

                                current_data = ""

                                async for line in response.aiter_lines():
                                    if line.startswith("data:"):
                                        payload = line[5:].strip()
                                        if not payload: continue
                                        if payload == "[DONE]":
                                            logger.info("收到[DONE]标记，结束处理")
                                            break
                                        current_data += payload
                                        continue

                                    if (line.strip() == "") and current_data:
                                        raw_bytes = _parse_payload_bytes2(current_data)
                                        current_data = ""
                                        if raw_bytes is None:
                                            logger.debug("跳过无法解析的SSE数据块（非hex/base64或不完整）")
                                            continue
                                        try:
                                            event_data = protobuf_to_dict(raw_bytes,
                                                                          "warp.multi_agent.v1.ResponseEvent")
                                            event_count += 1
                                            event_type = _get_event_type(event_data)
                                            parsed_event = {"event_number": event_count, "event_type": event_type,
                                                            "parsed_data": event_data}
                                            parsed_events.append(parsed_event)
                                            logger.info(f"🔄 Event #{event_count}: {event_type}")
                                            logger.debug(f"   📋 Event data: {str(event_data)}")

                                            def _get(d: Dict[str, Any], *names: str) -> Any:
                                                for n in names:
                                                    if isinstance(d, dict) and n in d:
                                                        return d[n]
                                                return None

                                            if "init" in event_data:
                                                init_data = event_data["init"]
                                                conversation_id = init_data.get("conversation_id", conversation_id)
                                                task_id = init_data.get("task_id", task_id)
                                                logger.info(f"会话初始化: {conversation_id}")

                                            client_actions = _get(event_data, "client_actions", "clientActions")
                                            if isinstance(client_actions, dict):
                                                actions = _get(client_actions, "actions", "Actions") or []
                                                for i, action in enumerate(actions):
                                                    logger.info(f"   🎯 Action #{i + 1}: {list(action.keys())}")

                                                    # 处理 update_task_message（新增）
                                                    update_msg_data = _get(action, "update_task_message",
                                                                           "updateTaskMessage")
                                                    if isinstance(update_msg_data, dict):
                                                        message = update_msg_data.get("message", {})
                                                        text_content = _extract_text_from_message(message)
                                                        if text_content:
                                                            complete_response.append(text_content)
                                                            logger.info(
                                                                f"   📝 Text from UPDATE_MESSAGE: {text_content}")

                                                    # 处理 append_to_message_content
                                                    append_data = _get(action, "append_to_message_content",
                                                                       "appendToMessageContent")
                                                    if isinstance(append_data, dict):
                                                        message = append_data.get("message", {})
                                                        agent_output = _get(message, "agent_output",
                                                                            "agentOutput") or {}
                                                        text_content = agent_output.get("text", "")
                                                        if text_content:
                                                            complete_response.append(text_content)
                                                            logger.info(f"   📝 Text Fragment: {text_content}")

                                                    # 处理 add_messages_to_task
                                                    messages_data = _get(action, "add_messages_to_task",
                                                                         "addMessagesToTask")
                                                    if isinstance(messages_data, dict):
                                                        messages = messages_data.get("messages", [])
                                                        task_id = messages_data.get("task_id",
                                                                                    messages_data.get("taskId",
                                                                                                      task_id))
                                                        for j, message in enumerate(messages):
                                                            logger.info(
                                                                f"   📨 Message #{j + 1}: {list(message.keys())}")
                                                            text_content = _extract_text_from_message(message)
                                                            if text_content:
                                                                complete_response.append(text_content)
                                                                logger.info(
                                                                    f"   📝 Complete Message: {text_content}")
                                        except Exception as parse_err:
                                            logger.debug(f"解析事件失败，跳过: {str(parse_err)}")
                                            continue

                                # 成功处理完响应，生成结果并返回
                                full_response = "".join(complete_response)
                                logger.info("=" * 60)
                                logger.info("📊 SSE STREAM SUMMARY (解析模式)")
                                logger.info("=" * 60)
                                logger.info(f"📈 Total Events Processed: {event_count}")
                                logger.info(f"🆔 Conversation ID: {conversation_id}")
                                logger.info(f"🆔 Task ID: {task_id}")
                                logger.info(f"📝 Response Length: {len(full_response)} characters")
                                logger.info(f"🎯 Parsed Events Count: {len(parsed_events)}")
                                logger.info("=" * 60)

                                # 成功完成，释放会话并返回结果
                                await release_pool_session(current_session.get("session_id"))
                                current_session = None

                                logger.info(f"✅ Stream processing completed successfully (解析模式)")
                                return full_response, conversation_id, task_id, parsed_events

                            # 错误处理（429等）
                            error_text = await response.aread()
                            error_content = error_text.decode('utf-8') if error_text else "No error content"

                            # 检查是否是账号被封禁错误 (403)
                            is_blocked_error = (
                                    response.status_code == 403 and (
                                    ("Your account has been blocked" in error_content) or
                                    ("blocked from using AI features" in error_content)
                            )
                            )

                            if is_blocked_error:
                                logger.error(f"❌ 账号 {account_email} 已被封禁 (HTTP 403, 解析模式)")
                                # 释放并标记为blocked
                                if current_session:
                                    # 通知pool service标记账号
                                    try:
                                        async with httpx.AsyncClient(timeout=5.0) as notify_client:
                                            await notify_client.post(
                                                "http://localhost:8019/api/accounts/mark_blocked",
                                                json={"email": account_email}
                                            )
                                    except:
                                        pass

                                    await release_pool_session(current_session.get("session_id"))
                                    current_session = None

                                # 如果还有重试次数，获取新账号
                                if attempt < (MAX_QUOTA_RETRIES - 1):
                                    logger.warning(
                                        f"账号被封(解析模式)，将获取新账号重试 (第 {attempt + 2}/{MAX_QUOTA_RETRIES} 次)...")
                                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                                    break  # 跳出代理循环，进入下一个attempt获取新账号
                                else:
                                    return f"❌ Account blocked after {MAX_QUOTA_RETRIES} attempts", None, None, []

                            is_quota_error = ("No remaining quota" in error_content) or (
                                    "No AI requests remaining" in error_content)

                            if response.status_code == 429 and is_quota_error:
                                if attempt < (MAX_QUOTA_RETRIES - 1):
                                    logger.warning(
                                        f"Warp API 返回 429 (配额用尽/解析模式)。将在 {RETRY_DELAY_SECONDS} 秒后强制获取新账号并重试 (第 {attempt + 2}/{MAX_QUOTA_RETRIES} 次)...")
                                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                                    # 跳出代理循环，进入下一个attempt获取新账号
                                    break
                                else:
                                    # 所有账号都用尽了
                                    await release_pool_session(current_session.get("session_id"))
                                    current_session = None
                                    return f"❌ API Error (HTTP {response.status_code}) after {MAX_QUOTA_RETRIES} attempts: {error_content}", None, None, []

                            # 其他HTTP错误，尝试换代理
                            logger.error(
                                f"HTTP错误 {response.status_code}(解析模式)，尝试换代理 (proxy attempt {proxy_attempt + 1}/{max_proxy_retries})")
                            if proxy_attempt < max_proxy_retries - 1:
                                await asyncio.sleep(0.5)
                                continue

                            if attempt < (MAX_QUOTA_RETRIES - 1):
                                logger.warning(f"当前账号的所有代理都失败(解析模式)，将换新账号重试")
                                break

                            # 真正失败了
                            await release_pool_session(current_session.get("session_id"))
                            current_session = None
                            return f"❌ API Error (HTTP {response.status_code}): {error_content}", None, None, []

                except (httpx.ConnectError, httpx.ProxyError, httpx.RemoteProtocolError) as ssl_error:
                    logger.warning(
                        f"SSL/代理错误(解析模式) (proxy attempt {proxy_attempt + 1}/{max_proxy_retries}): {ssl_error}")
                    if proxy_attempt < max_proxy_retries - 1:
                        await asyncio.sleep(0.5)
                        continue
                    # 所有代理都失败，进入下一个attempt
                    break

                except httpx.ReadTimeout:
                    logger.warning(
                        f"请求超时(解析模式)，尝试换代理 (proxy attempt {proxy_attempt + 1}/{max_proxy_retries})")
                    if proxy_attempt < max_proxy_retries - 1:
                        await asyncio.sleep(0.5)
                        continue
                    break

                except Exception as e:
                    logger.error(f"未知错误(解析模式): {e}")
                    if proxy_attempt < max_proxy_retries - 1:
                        await asyncio.sleep(0.5)
                        continue
                    raise

        # ⚠️ 新增：所有重试都失败后的默认返回
        logger.error(f"所有 {MAX_QUOTA_RETRIES} 次重试都失败了(解析模式)")
        if current_session:
            await release_pool_session(current_session.get("session_id"))
            current_session = None
        return "❌ All retry attempts failed", None, None, []

    except Exception as e:
        import traceback
        logger.error("=" * 60)
        logger.error("WARP API CLIENT EXCEPTION (解析模式)")
        logger.error("=" * 60)
        logger.error(f"Exception Type: {type(e).__name__}")
        logger.error(f"Exception Message: {str(e)}")
        logger.error(f"Request URL: {warp_url if 'warp_url' in locals() else 'Unknown'}")
        logger.error(f"Request Size: {len(protobuf_bytes) if 'protobuf_bytes' in locals() else 'Unknown'}")
        logger.error("Python Traceback:")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
        # ⚠️ 新增：异常时也返回正确格式
        if current_session:
            await release_pool_session(current_session.get("session_id"))
        return f"❌ Exception: {str(e)}", None, None, []
    finally:
        # 确保会话被释放
        if current_session:
            await release_pool_session(current_session.get("session_id"))
