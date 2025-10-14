#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protobuf编解码API路由

提供纯protobuf数据包编解码服务，包括JWT管理和WebSocket支持。
"""
import base64
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..config.settings import CLIENT_VERSION, OS_CATEGORY, OS_NAME, OS_VERSION, WARP_URL as CONFIG_WARP_URL
from ..core.auth import get_jwt_token, is_token_expired, refresh_jwt_if_needed, get_valid_jwt
from ..core.logging import logger
from ..core.pool_auth import acquire_pool_or_anonymous_token
from ..core.protobuf_utils import protobuf_to_dict, dict_to_protobuf_bytes
from ..core.server_message_data import decode_server_message_data, encode_server_message_data
from ..core.stream_processor import set_websocket_manager


def _encode_smd_inplace(obj: Any) -> Any:
    if isinstance(obj, dict):
        new_d = {}
        for k, v in obj.items():
            if k in ("server_message_data", "serverMessageData") and isinstance(v, dict):
                try:
                    b64 = encode_server_message_data(
                        uuid=v.get("uuid"),
                        seconds=v.get("seconds"),
                        nanos=v.get("nanos"),
                    )
                    new_d[k] = b64
                except Exception:
                    new_d[k] = v
            else:
                new_d[k] = _encode_smd_inplace(v)
        return new_d
    elif isinstance(obj, list):
        return [_encode_smd_inplace(x) for x in obj]
    else:
        return obj


def _decode_smd_inplace(obj: Any) -> Any:
    if isinstance(obj, dict):
        new_d = {}
        for k, v in obj.items():
            if k in ("server_message_data", "serverMessageData") and isinstance(v, str):
                try:
                    dec = decode_server_message_data(v)
                    new_d[k] = dec
                except Exception:
                    new_d[k] = v
            else:
                new_d[k] = _decode_smd_inplace(v)
        return new_d
    elif isinstance(obj, list):
        return [_decode_smd_inplace(x) for x in obj]
    else:
        return obj
from ..core.schema_sanitizer import sanitize_mcp_input_schema_in_packet


class EncodeRequest(BaseModel):
    json_data: Optional[Dict[str, Any]] = None
    message_type: str = "warp.multi_agent.v1.Request"

    task_context: Optional[Dict[str, Any]] = None
    input: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    mcp_context: Optional[Dict[str, Any]] = None
    existing_suggestions: Optional[Dict[str, Any]] = None
    client_version: Optional[str] = None
    os_category: Optional[str] = None
    os_name: Optional[str] = None
    os_version: Optional[str] = None

    class Config:
        extra = "allow"

    def get_data(self) -> Dict[str, Any]:
        if self.json_data is not None:
            return self.json_data
        else:
            data: Dict[str, Any] = {}
            if self.task_context is not None:
                data["task_context"] = self.task_context
            if self.input is not None:
                data["input"] = self.input
            if self.settings is not None:
                data["settings"] = self.settings
            if self.metadata is not None:
                data["metadata"] = self.metadata
            if self.mcp_context is not None:
                data["mcp_context"] = self.mcp_context
            if self.existing_suggestions is not None:
                data["existing_suggestions"] = self.existing_suggestions
            if self.client_version is not None:
                data["client_version"] = self.client_version
            if self.os_category is not None:
                data["os_category"] = self.os_category
            if self.os_name is not None:
                data["os_name"] = self.os_name
            if self.os_version is not None:
                data["os_version"] = self.os_version

            skip_keys = {
                "json_data", "message_type", "task_context", "input", "settings", "metadata",
                "mcp_context", "existing_suggestions", "client_version", "os_category", "os_name", "os_version"
            }
            try:
                for k, v in self.__dict__.items():
                    if v is None:
                        continue
                    if k in skip_keys:
                        continue
                    if k not in data:
                        data[k] = v
            except Exception:
                pass
            return data


class DecodeRequest(BaseModel):
    protobuf_bytes: str
    message_type: str = "warp.multi_agent.v1.Request"


class StreamDecodeRequest(BaseModel):
    protobuf_chunks: List[str]
    message_type: str = "warp.multi_agent.v1.Response"


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.packet_history: List[Dict] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket连接建立，当前连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket连接断开，当前连接数: {len(self.active_connections)}")

    async def broadcast(self, message: Dict):
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"发送WebSocket消息失败: {e}")
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

    async def log_packet(self, packet_type: str, data: Dict, size: int):
        packet_info = {
            "timestamp": datetime.now().isoformat(),
            "type": packet_type,
            "size": size,
            "data_preview": str(data)[:200] + "..." if len(str(data)) > 200 else str(data),
            "full_data": data
        }

        self.packet_history.append(packet_info)
        if len(self.packet_history) > 100:
            self.packet_history = self.packet_history[-100:]

        await self.broadcast({"event": "packet_captured", "packet": packet_info})


manager = ConnectionManager()
set_websocket_manager(manager)

app = FastAPI(title="Protobuf 编解码服务器", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Protobuf 编解码服务器", "version": "1.0.0"}


@app.get("/healthz")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/encode")
async def encode_json_to_protobuf(request: EncodeRequest):
    try:
        logger.info(f"收到编码请求，消息类型: {request.message_type}")
        actual_data = request.get_data()
        if not actual_data:
            raise HTTPException(400, "数据包不能为空")
        wrapped = {"json_data": actual_data}
        wrapped = sanitize_mcp_input_schema_in_packet(wrapped)
        actual_data = wrapped.get("json_data", actual_data)
        actual_data = _encode_smd_inplace(actual_data)
        protobuf_bytes = dict_to_protobuf_bytes(actual_data, request.message_type)
        try:
            await manager.log_packet("encode", actual_data, len(protobuf_bytes))
        except Exception as log_error:
            logger.warning(f"数据包记录失败: {log_error}")
        result = {
            "protobuf_bytes": base64.b64encode(protobuf_bytes).decode('utf-8'),
            "size": len(protobuf_bytes),
            "message_type": request.message_type
        }
        logger.info(f"✅ JSON编码为protobuf成功: {len(protobuf_bytes)} 字节")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ JSON编码失败: {e}")
        raise HTTPException(500, f"编码失败: {str(e)}")


@app.post("/api/decode")
async def decode_protobuf_to_json(request: DecodeRequest):
    try:
        logger.info(f"收到解码请求，消息类型: {request.message_type}")
        if not request.protobuf_bytes or not request.protobuf_bytes.strip():
            raise HTTPException(400, "Protobuf数据不能为空")
        try:
            protobuf_bytes = base64.b64decode(request.protobuf_bytes)
        except Exception as decode_error:
            logger.error(f"Base64解码失败: {decode_error}")
            raise HTTPException(400, f"Base64解码失败: {str(decode_error)}")
        if not protobuf_bytes:
            raise HTTPException(400, "解码后的protobuf数据为空")
        json_data = protobuf_to_dict(protobuf_bytes, request.message_type)
        try:
            await manager.log_packet("decode", json_data, len(protobuf_bytes))
        except Exception as log_error:
            logger.warning(f"数据包记录失败: {log_error}")
        result = {"json_data": json_data, "size": len(protobuf_bytes), "message_type": request.message_type}
        logger.info(f"✅ Protobuf解码为JSON成功: {len(protobuf_bytes)} 字节")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Protobuf解码失败: {e}")
        raise HTTPException(500, f"解码失败: {e}")


@app.post("/api/stream-decode")
async def decode_stream_protobuf(request: StreamDecodeRequest):
    try:
        logger.info(f"收到流式解码请求，数据块数量: {len(request.protobuf_chunks)}")
        results = []
        total_size = 0
        for i, chunk_b64 in enumerate(request.protobuf_chunks):
            try:
                chunk_bytes = base64.b64decode(chunk_b64)
                chunk_json = protobuf_to_dict(chunk_bytes, request.message_type)
                chunk_result = {"chunk_index": i, "json_data": chunk_json, "size": len(chunk_bytes)}
                results.append(chunk_result)
                total_size += len(chunk_bytes)
                await manager.log_packet(f"stream_decode_chunk_{i}", chunk_json, len(chunk_bytes))
            except Exception as e:
                logger.warning(f"数据块 {i} 解码失败: {e}")
                results.append({"chunk_index": i, "error": str(e), "size": 0})
        try:
            all_bytes = b''.join([base64.b64decode(chunk) for chunk in request.protobuf_chunks])
            complete_json = protobuf_to_dict(all_bytes, request.message_type)
            await manager.log_packet("stream_decode_complete", complete_json, len(all_bytes))
            complete_result = {"json_data": complete_json, "size": len(all_bytes)}
        except Exception as e:
            complete_result = {"error": f"无法拼接完整消息: {e}", "size": total_size}
        result = {"chunks": results, "complete": complete_result, "total_chunks": len(request.protobuf_chunks), "total_size": total_size, "message_type": request.message_type}
        logger.info(f"✅ 流式protobuf解码完成: {len(request.protobuf_chunks)} 块，总大小 {total_size} 字节")
        return result
    except Exception as e:
        logger.error(f"❌ 流式protobuf解码失败: {e}")
        raise HTTPException(500, f"流式解码失败: {e}")


@app.get("/api/schemas")
async def get_protobuf_schemas():
    try:
        from ..core.protobuf import ensure_proto_runtime, ALL_MSGS, msg_cls
        ensure_proto_runtime()
        schemas = []
        for msg_name in ALL_MSGS:
            try:
                MessageClass = msg_cls(msg_name)
                descriptor = MessageClass.DESCRIPTOR
                fields = []
                for field in descriptor.fields:
                    fields.append({"name": field.name, "type": field.type, "label": getattr(field, 'label', None), "number": field.number})
                schemas.append({"name": msg_name, "full_name": descriptor.full_name, "field_count": len(fields), "fields": fields[:10]})
            except Exception as e:
                logger.warning(f"获取schema {msg_name} 信息失败: {e}")
        result = {"schemas": schemas, "total_count": len(schemas), "message": f"找到 {len(schemas)} 个protobuf消息类型"}
        logger.info(f"✅ 返回 {len(schemas)} 个protobuf schema")
        return result
    except Exception as e:
        logger.error(f"❌ 获取protobuf schemas失败: {e}")
        raise HTTPException(500, f"获取schemas失败: {e}")


@app.get("/api/auth/status")
async def get_auth_status():
    try:
        jwt_token = get_jwt_token()
        if not jwt_token:
            return {"authenticated": False, "message": "未找到JWT token", "suggestion": "运行 'uv run refresh_jwt.py' 获取token"}
        is_expired = is_token_expired(jwt_token)
        result = {"authenticated": not is_expired, "token_present": True, "token_expired": is_expired, "token_preview": f"{jwt_token[:20]}...{jwt_token[-10:]}", "message": "Token有效" if not is_expired else "Token已过期"}
        if is_expired:
            result["suggestion"] = "运行 'uv run refresh_jwt.py' 刷新token"
        return result
    except Exception as e:
        logger.error(f"❌ 获取认证状态失败: {e}")
        raise HTTPException(500, f"获取认证状态失败: {e}")


@app.post("/api/auth/refresh")
async def refresh_auth_token():
    try:
        success = await refresh_jwt_if_needed()
        if success:
            return {"success": True, "message": "JWT token刷新成功", "timestamp": datetime.now().isoformat()}
        else:
            return {"success": False, "message": "JWT token刷新失败", "suggestion": "检查网络连接或手动运行 'uv run refresh_jwt.py'"}
    except Exception as e:
        logger.error(f"❌ 刷新JWT token失败: {e}")
        raise HTTPException(500, f"刷新token失败: {e}")


@app.get("/api/auth/user_id")
async def get_user_id_endpoint():
    try:
        from ..core.auth import get_user_id
        user_id = get_user_id()
        if user_id:
            return {"success": True, "user_id": user_id, "message": "User ID获取成功"}
        else:
            return {"success": False, "user_id": "", "message": "未找到User ID，可能需要刷新JWT token"}
    except Exception as e:
        logger.error(f"❌ 获取User ID失败: {e}")
        raise HTTPException(500, f"获取User ID失败: {e}")


@app.get("/api/packets/history")
async def get_packet_history(limit: int = 50):
    try:
        history = manager.packet_history[-limit:] if len(manager.packet_history) > limit else manager.packet_history
        return {"packets": history, "total_count": len(manager.packet_history), "returned_count": len(history)}
    except Exception as e:
        logger.error(f"❌ 获取数据包历史失败: {e}")
        raise HTTPException(500, f"获取历史记录失败: {e}")


@app.post("/api/warp/send")
async def send_to_warp_api(
    request: EncodeRequest,
    show_all_events: bool = Query(True, description="Show detailed SSE event breakdown")
):
    try:
        logger.info(f"收到Warp API发送请求，消息类型: {request.message_type}")
        actual_data = request.get_data()
        if not actual_data:
            raise HTTPException(400, "数据包不能为空")
        wrapped = {"json_data": actual_data}
        wrapped = sanitize_mcp_input_schema_in_packet(wrapped)
        actual_data = wrapped.get("json_data", actual_data)
        actual_data = _encode_smd_inplace(actual_data)
        protobuf_bytes = dict_to_protobuf_bytes(actual_data, request.message_type)
        logger.info(f"✅ JSON编码为protobuf成功: {len(protobuf_bytes)} 字节")
        from ..warp.api_client import send_protobuf_to_warp_api
        response_text, conversation_id, task_id = await send_protobuf_to_warp_api(protobuf_bytes, show_all_events=show_all_events)
        await manager.log_packet("warp_request", actual_data, len(protobuf_bytes))
        await manager.log_packet("warp_response", {"response": response_text, "conversation_id": conversation_id, "task_id": task_id}, len(response_text.encode()))
        result = {"response": response_text, "conversation_id": conversation_id, "task_id": task_id, "request_size": len(protobuf_bytes), "response_size": len(response_text), "message_type": request.message_type}
        logger.info(f"✅ Warp API调用成功，响应长度: {len(response_text)} 字符")
        return result
    except Exception as e:
        import traceback
        error_details = {"error": str(e), "error_type": type(e).__name__, "traceback": traceback.format_exc(), "request_info": {"message_type": request.message_type, "json_size": len(str(actual_data)), "has_tools": "mcp_context" in actual_data, "has_history": "task_context" in actual_data}}
        logger.error(f"❌ Warp API调用失败: {e}")
        logger.error(f"错误详情: {error_details}")
        try:
            await manager.log_packet("warp_error", error_details, 0)
        except Exception as log_error:
            logger.warning(f"记录错误失败: {log_error}")
        raise HTTPException(500, detail=error_details)


@app.post("/api/warp/send_stream")
async def send_to_warp_api_parsed(
    request: EncodeRequest
):
    try:
        logger.info(f"收到Warp API解析发送请求，消息类型: {request.message_type}")
        actual_data = request.get_data()
        if not actual_data:
            raise HTTPException(400, "数据包不能为空")
        wrapped = {"json_data": actual_data}
        wrapped = sanitize_mcp_input_schema_in_packet(wrapped)
        actual_data = wrapped.get("json_data", actual_data)
        actual_data = _encode_smd_inplace(actual_data)
        protobuf_bytes = dict_to_protobuf_bytes(actual_data, request.message_type)
        logger.info(f"✅ JSON编码为protobuf成功: {len(protobuf_bytes)} 字节")
        from ..warp.api_client import send_protobuf_to_warp_api_parsed
        response_text, conversation_id, task_id, parsed_events = await send_protobuf_to_warp_api_parsed(protobuf_bytes)
        parsed_events = _decode_smd_inplace(parsed_events)
        await manager.log_packet("warp_request_parsed", actual_data, len(protobuf_bytes))
        response_data = {"response": response_text, "conversation_id": conversation_id, "task_id": task_id, "parsed_events": parsed_events}
        await manager.log_packet("warp_response_parsed", response_data, len(str(response_data)))
        result = {"response": response_text, "conversation_id": conversation_id, "task_id": task_id, "request_size": len(protobuf_bytes), "response_size": len(response_text), "message_type": request.message_type, "parsed_events": parsed_events, "events_count": len(parsed_events), "events_summary": {}}
        if parsed_events:
            event_type_counts = {}
            for event in parsed_events:
                event_type = event.get("event_type", "UNKNOWN")
                event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            result["events_summary"] = event_type_counts
        logger.info(f"✅ Warp API解析调用成功，响应长度: {len(response_text)} 字符，事件数量: {len(parsed_events)}")
        return result
    except Exception as e:
        import traceback
        error_details = {"error": str(e), "error_type": type(e).__name__, "traceback": traceback.format_exc(), "request_info": {"message_type": request.message_type, "json_size": len(str(actual_data)) if 'actual_data' in locals() else 0, "has_tools": "mcp_context" in (actual_data or {}), "has_history": "task_context" in (actual_data or {})}}
        logger.error(f"❌ Warp API解析调用失败: {e}")
        logger.error(f"错误详情: {error_details}")
        try:
            await manager.log_packet("warp_error_parsed", error_details, 0)
        except Exception as log_error:
            logger.warning(f"记录错误失败: {log_error}")
        raise HTTPException(500, detail=error_details)


@app.post("/api/warp/send_stream_sse")
async def send_to_warp_api_stream_sse(request: EncodeRequest):
    from fastapi.responses import StreamingResponse
    import os as _os
    import re as _re
    import asyncio
    # 导入代理管理器
    from ..core.proxy_manager import AsyncProxyManager

    try:
        actual_data = request.get_data()
        if not actual_data:
            raise HTTPException(400, "数据包不能为空")
        wrapped = {"json_data": actual_data}
        wrapped = sanitize_mcp_input_schema_in_packet(wrapped)
        actual_data = wrapped.get("json_data", actual_data)
        actual_data = _encode_smd_inplace(actual_data)
        protobuf_bytes = dict_to_protobuf_bytes(actual_data, request.message_type)

        async def _agen():
            # 创建代理管理器实例
            proxy_manager = AsyncProxyManager()
            max_proxy_retries = 7  # 增加到 7 次代理重试
            max_attempts = 5

            warp_url = CONFIG_WARP_URL

            def _parse_payload_bytes(data_str: str):
                s = _re.sub(r"\\s+", "", data_str or "")
                if not s:
                    return None
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

            verify_opt = False  # 使用代理时关闭SSL验证
            insecure_env = _os.getenv("WARP_INSECURE_TLS", "").lower()
            if insecure_env in ("1", "true", "yes"):
                verify_opt = False
                logger.warning("TLS verification disabled via WARP_INSECURE_TLS for Warp API stream endpoint")

            # 最多尝试四次：第一次失败且为配额429时申请匿名token并重试
            jwt = None
            successful = False
            last_error = None

            for attempt in range(max_attempts):
                if attempt > 0:
                    logger.info(f"开始第 {attempt + 1}/{max_attempts} 轮总体重试...")
                    # 指数退避：2秒、4秒、8秒
                    await asyncio.sleep(2.0 ** attempt)

                for proxy_attempt in range(max_proxy_retries):
                    try:
                        # 获取新的代理
                        proxy_str = await proxy_manager.get_proxy()
                        proxy_config = None

                        if proxy_str:
                            proxy_config = proxy_manager.format_proxy_for_httpx(proxy_str)

                        # 创建带代理的客户端配置
                        client_config = {
                            "http2": True,
                            "timeout": httpx.Timeout(
                                timeout=600.0,
                                connect=15.0,  # 连接超时15秒
                                read=120.0,  # 读取超时120秒
                                write=15.0,  # 写入超时15秒
                                pool=15.0  # 连接池超时15秒
                            ),
                            "verify": verify_opt,
                            "trust_env": False,  # 禁用环境代理，完全使用代码控制
                            "limits": httpx.Limits(
                                max_keepalive_connections=10,
                                max_connections=20,
                                keepalive_expiry=60
                            )
                        }

                        # 如果有代理配置，添加代理参数
                        if proxy_config:
                            client_config["proxy"] = proxy_config

                        async with httpx.AsyncClient(**client_config) as client:
                            if attempt == 0 or jwt is None:
                                jwt = await get_valid_jwt()

                            headers = {
                                "accept": "text/event-stream",
                                "content-type": "application/x-protobuf",
                                "x-warp-client-version": CLIENT_VERSION,
                                "x-warp-os-category": OS_CATEGORY,
                                "x-warp-os-name": OS_NAME,
                                "x-warp-os-version": OS_VERSION,
                                "authorization": f"Bearer {jwt}",
                                "content-length": str(len(protobuf_bytes)),
                            }

                            async with client.stream("POST", warp_url, headers=headers,
                                                     content=protobuf_bytes) as response:
                                if response.status_code != 200:
                                    error_text = await response.aread()
                                    error_content = error_text.decode("utf-8") if error_text else ""

                                    # 检查是否是账号被封禁 (403)
                                    if response.status_code == 403 and (
                                            ("Your account has been blocked" in error_content) or
                                            ("blocked from using AI features" in error_content)
                                    ):
                                        logger.error(
                                            f"❌ 账号已被封禁 (HTTP 403, attempt {attempt + 1})。立即删除并获取新账号..."
                                        )

                                        # 标记当前账号为blocked（如果有pool service）
                                        if jwt:
                                            try:
                                                # 通知账号池服务该账号已被封（本地服务，禁用环境代理）
                                                async with httpx.AsyncClient(timeout=5.0, trust_env=False, proxy=None) as notify_client:
                                                    await notify_client.post(
                                                        "http://127.0.0.1:8019/api/accounts/mark_blocked",
                                                        json={"jwt_token": jwt[:50]}  # 只传部分token作为标识
                                                    )
                                            except Exception as e:
                                                logger.warning(f"无法通知账号池服务: {e}")

                                        # 强制获取新账号，不再使用当前账号
                                        try:
                                            new_jwt = await acquire_pool_or_anonymous_token(force_new=True)
                                            if new_jwt:
                                                jwt = new_jwt
                                                logger.info("✅ 获取新账号token成功（账号被封后）")
                                                # 跳出proxy循环，进入下一个attempt
                                                break
                                        except Exception as e:
                                            logger.error(f"获取新账号失败: {e}")

                                        # 如果无法获取新账号或已是最后一次尝试，返回错误
                                        if attempt >= max_attempts - 1:
                                            yield f"data: {{\"error\": \"Account blocked and unable to get new account\"}}\\n\\n"
                                            yield "data: [DONE]\\n\\n"
                                            return
                                        else:
                                            break  # 跳出proxy循环，用新账号重试

                                    # 429 且包含配额信息时，申请匿名token后重试
                                    elif response.status_code == 429 and (
                                            ("No remaining quota" in error_content) or
                                            ("No AI requests remaining" in error_content)
                                    ):
                                        logger.warning(
                                            f"Warp API 返回 429 (额度用尽, SSE 代理, attempt {attempt + 1})。尝试强制获取新账号token...")
                                        try:
                                            # force_new=True 强制获取新账号
                                            new_jwt = await acquire_pool_or_anonymous_token(force_new=True)
                                            if new_jwt:
                                                jwt = new_jwt
                                                logger.info("✅ 获取新账号token成功，将在下一轮重试")
                                                # 跳出proxy循环，进入下一个attempt
                                                break
                                        except Exception as e:
                                            logger.error(f"获取新token失败: {e}")

                                    # 其他HTTP错误，记录并继续尝试
                                    logger.error(
                                        f"Warp API HTTP error {response.status_code} (attempt {attempt + 1}/{max_attempts}, proxy {proxy_attempt + 1}/{max_proxy_retries}): {error_content[:300]}")
                                    last_error = f"HTTP {response.status_code}: {error_content[:100]}"

                                    if proxy_attempt < max_proxy_retries - 1:
                                        continue  # 继续下一个proxy_attempt

                                    # 当前attempt的所有代理都失败，准备下一轮
                                    if attempt < max_attempts - 1:
                                        logger.info(f"第 {attempt + 1} 轮所有代理失败，准备下一轮...")
                                        break  # 跳出proxy循环

                                    # 真正失败了，返回错误
                                    yield f"data: {{\"error\": \"HTTP {response.status_code} after {max_attempts} attempts\"}}\n\n"
                                    yield "data: [DONE]\n\n"
                                    return

                                # 请求成功，处理SSE流
                                try:
                                    logger.info(f"✅ Warp API SSE连接已建立: {warp_url}")
                                    logger.info(f"📦 请求字节数: {len(protobuf_bytes)}")
                                    logger.info(f"🔄 使用代理: {proxy_config if proxy_config else '直连'}")
                                    logger.info(
                                        f"🔢 尝试次数: attempt={attempt + 1}/{max_attempts}, proxy={proxy_attempt + 1}/{max_proxy_retries}")
                                except Exception:
                                    pass

                                current_data = ""
                                event_no = 0
                                has_events = False

                                async for line in response.aiter_lines():
                                    if line.startswith("data:"):
                                        payload = line[5:].strip()
                                        if not payload:
                                            continue
                                        if payload == "[DONE]":
                                            successful = True
                                            break
                                        current_data += payload
                                        continue

                                    if (line.strip() == "") and current_data:
                                        raw_bytes = _parse_payload_bytes(current_data)
                                        current_data = ""
                                        if raw_bytes is None:
                                            continue

                                        try:
                                            event_data = protobuf_to_dict(raw_bytes,
                                                                          "warp.multi_agent.v1.ResponseEvent")
                                            has_events = True
                                        except Exception:
                                            continue

                                        def _get(d: Dict[str, Any], *names: str) -> Any:
                                            for n in names:
                                                if isinstance(d, dict) and n in d:
                                                    return d[n]
                                            return None

                                        event_type = "UNKNOWN_EVENT"
                                        if isinstance(event_data, dict):
                                            if "init" in event_data:
                                                event_type = "INITIALIZATION"
                                            else:
                                                client_actions = _get(event_data, "client_actions", "clientActions")
                                                if isinstance(client_actions, dict):
                                                    actions = _get(client_actions, "actions", "Actions") or []
                                                    event_type = f"CLIENT_ACTIONS({len(actions)})" if actions else "CLIENT_ACTIONS_EMPTY"
                                                elif "finished" in event_data:
                                                    event_type = "FINISHED"

                                        event_no += 1
                                        try:
                                            logger.info(f"🔄 SSE Event #{event_no}: {event_type} ---- {event_data}")
                                        except Exception:
                                            pass

                                        out = {"event_number": event_no, "event_type": event_type,
                                               "parsed_data": event_data}
                                        try:
                                            chunk = json.dumps(out, ensure_ascii=False)
                                        except Exception:
                                            logger.error(f"无法将事件数据转换为JSON: {out}")
                                            continue

                                        yield f"data: {chunk}\n\n"

                                # 检查是否成功接收到事件
                                if has_events or successful:
                                    try:
                                        logger.info("=" * 60)
                                        logger.info("📊 SSE STREAM SUMMARY (代理)")
                                        logger.info("=" * 60)
                                        logger.info(f"📈 Total Events Forwarded: {event_no}")
                                        logger.info(f"🔄 使用代理: {proxy_config if proxy_config else '直连'}")
                                        logger.info(
                                            f"✅ 成功完成 (attempt {attempt + 1}/{max_attempts}, proxy {proxy_attempt + 1}/{max_proxy_retries})")
                                        logger.info("=" * 60)
                                    except Exception:
                                        pass

                                    yield "data: [DONE]\n\n"
                                    return  # 成功完成，直接返回
                                else:
                                    # 没有收到任何事件，视为失败
                                    logger.warning(
                                        f"未收到任何事件，视为失败 (attempt {attempt + 1}/{max_attempts}, proxy {proxy_attempt + 1}/{max_proxy_retries})")
                                    last_error = "No events received"
                                    if proxy_attempt < max_proxy_retries - 1:
                                        continue

                    except (httpx.ConnectError, httpx.ProxyError, httpx.RemoteProtocolError) as ssl_error:
                        last_error = f"SSL/Proxy error: {str(ssl_error)}"
                        logger.warning(
                            f"SSE端点 SSL/代理错误 (attempt {attempt + 1}/{max_attempts}, proxy {proxy_attempt + 1}/{max_proxy_retries}): {ssl_error}"
                        )
                        if proxy_attempt < max_proxy_retries - 1:
                            continue  # 继续下一个proxy_attempt

                        # 当前attempt的所有代理都失败
                        if attempt < max_attempts - 1:
                            logger.info(f"第 {attempt + 1} 轮所有代理因SSL/代理错误失败，尝试获取新token...")
                            try:
                                new_jwt = await acquire_pool_or_anonymous_token()
                                if new_jwt:
                                    jwt = new_jwt
                                    logger.info("获取新token成功，将在下一轮重试")
                            except Exception as token_error:
                                logger.error(f"获取新token失败: {token_error}")
                            break  # 跳出proxy循环，进入下一个attempt

                    except httpx.ReadTimeout as timeout_error:
                        last_error = f"Timeout: {str(timeout_error)}"
                        logger.warning(
                            f"SSE端点超时 (attempt {attempt + 1}/{max_attempts}, proxy {proxy_attempt + 1}/{max_proxy_retries}): {last_error}"
                        )
                        if proxy_attempt < max_proxy_retries - 1:
                            continue

                    except httpx.WriteTimeout as write_timeout:
                        last_error = f"Write timeout: {str(write_timeout)}"
                        logger.warning(
                            f"SSE端点写入超时 (attempt {attempt + 1}/{max_attempts}, proxy {proxy_attempt + 1}/{max_proxy_retries}): {last_error}"
                        )
                        if proxy_attempt < max_proxy_retries - 1:
                            continue

                    except Exception as e:
                        last_error = f"Unknown error: {str(e)}"
                        logger.error(
                            f"SSE端点未知错误 (attempt {attempt + 1}/{max_attempts}, proxy {proxy_attempt + 1}/{max_proxy_retries}): {e}",
                            exc_info=True)
                        if proxy_attempt < max_proxy_retries - 1:
                            continue

            # 所有尝试都失败了
            logger.error(f"SSE端点在 {max_attempts} 轮尝试（每轮 {max_proxy_retries} 个代理）后完全失败")
            yield f"data: {{\"error\": \"All {max_attempts} attempts failed. Last error: {last_error}\"}}\n\n"
            yield "data: [DONE]\n\n"
            return

        return StreamingResponse(_agen(), media_type="text/event-stream",
                                 headers={
                                     "Cache-Control": "no-cache",
                                     "Connection": "keep-alive",
                                     "X-Accel-Buffering": "no"  # 禁用nginx缓冲
                                 })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = {"error": str(e), "error_type": type(e).__name__, "traceback": traceback.format_exc()}
        logger.error(f"Warp SSE转发端点错误: {e}")
        raise HTTPException(500, detail=error_details)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({"event": "connected", "message": "WebSocket连接已建立", "timestamp": datetime.now().isoformat()})
        recent_packets = manager.packet_history[-10:]
        for packet in recent_packets:
            await websocket.send_json({"event": "packet_history", "packet": packet})
        while True:
            data = await websocket.receive_text()
            logger.debug(f"收到WebSocket消息: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
