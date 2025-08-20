import json
import base64
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from managers import tts_manager, websocket_manager, tts_config, audio_config, server_config, logger

router = APIRouter()

@router.websocket("/ws/tts")
async def websocket_endpoint(websocket: WebSocket):
    connection_id = str(uuid.uuid4())
    try:
        await websocket_manager.connect(websocket, connection_id)
        success = await websocket_manager.safe_send_text(websocket, connection_id, json.dumps({
            "status": "connected",
            "message": "TTS WebSocket connected successfully"
        }))
        if not success:
            logger.warning(f"Failed to send initial connection message to {connection_id}")
            return
        await websocket_manager.start_bidirectional_streaming(websocket, connection_id)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        websocket_manager.disconnect(connection_id)
        try:
            await websocket.close()
        except:
            pass

@router.get("/")
async def root():
    try:
        return FileResponse("frontend/index.html")
    except FileNotFoundError:
        try:
            return FileResponse("../frontend/index.html")
        except FileNotFoundError:
            return {
                "message": "SigIQ TTS WebSocket API",
                "endpoints": {
                    "websocket": "/ws/tts",
                    "docs": "/docs"
                },
                "note": "Frontend files not found. Check if frontend/index.html exists."
            }

@router.get("/frontend")
async def serve_frontend():
    try:
        return FileResponse("frontend/index.html")
    except FileNotFoundError:
        try:
            return FileResponse("../frontend/index.html")
        except FileNotFoundError:
            return {"error": "Frontend files not found"}

@router.get("/test")
async def serve_test():
    try:
        return FileResponse("test_tts.html")
    except FileNotFoundError:
        try:
            return FileResponse("../test_tts.html")
        except FileNotFoundError:
            return {"error": "Test file not found"}

@router.get("/test_tts.html")
async def serve_test_html():
    try:
        return FileResponse("test_tts.html")
    except FileNotFoundError:
        try:
            return FileResponse("../test_tts.html")
        except FileNotFoundError:
            return {"error": "Test HTML file not found"}

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "tts_engine": tts_manager.get_model_info(),
        "active_connections": len(websocket_manager.active_connections)
    }

@router.get("/models")
async def get_models():
    return {
        "available_models": ["kokoro", "chatterbox"],
        "current_model": tts_manager.get_model_info(),
        "configuration": {
            "kokoro": tts_config.get("kokoro", {}),
            "chatterbox": tts_config.get("chatterbox", {})
        }
    }

@router.post("/models/switch/{model_type}")
async def switch_model(model_type: str):
    if model_type not in ["kokoro", "chatterbox"]:
        return {"error": f"Unsupported model type: {model_type}. Supported types: kokoro, chatterbox"}
    success = tts_manager.switch_model(model_type)
    if success:
        return {
            "message": f"Successfully switched to {model_type} model",
            "current_model": tts_manager.get_model_info()
        }
    else:
        return {"error": f"Failed to switch to {model_type} model"}

@router.post("/models/refresh")
async def refresh_model_configuration():
    try:
        refreshed = tts_manager.refresh_configuration()
        if refreshed:
            return {
                "message": "Configuration refreshed and model reinitialized",
                "current_model": tts_manager.get_model_info()
            }
        else:
            return {
                "message": "No configuration changes detected",
                "current_model": tts_manager.get_model_info()
            }
    except Exception as e:
        return {"error": f"Failed to refresh configuration: {str(e)}"}

@router.get("/models/current")
async def get_current_model():
    return tts_manager.get_model_info()

@router.get("/models/debug")
async def debug_model_state():
    return {
        "model_type": tts_manager.model_type,
        "model_object": str(type(tts_manager.model)) if tts_manager.model else "None",
        "device": tts_manager.device,
        "configuration": {
            "selected_model": tts_config.get("selected_model"),
            "kokoro_config": tts_config.get("kokoro", {}),
            "chatterbox_config": tts_config.get("chatterbox", {})
        }
    }

@router.get("/models/kokoro/check")
async def check_kokoro_availability():
    is_available = tts_manager.check_kokoro_availability()
    return {
        "kokoro_available": is_available,
        "message": "Kokoro TTS is available and working" if is_available else "Kokoro TTS is not available or has issues"
    }

@router.get("/connections/{connection_id}/buffer")
async def get_connection_buffer(connection_id: str):
    if connection_id in websocket_manager.connection_buffers:
        buffer_content = websocket_manager.get_buffer_content(connection_id)
        return {
            "connection_id": connection_id,
            "buffer_content": buffer_content,
            "buffer_length": len(buffer_content),
            "is_connected": websocket_manager.is_connected(connection_id)
        }
    else:
        return {"error": f"Connection {connection_id} not found"}

@router.post("/connections/{connection_id}/buffer/clear")
async def clear_connection_buffer(connection_id: str):
    if connection_id in websocket_manager.connection_buffers:
        websocket_manager.clear_buffer(connection_id)
        return {
            "message": f"Buffer cleared for connection {connection_id}",
            "connection_id": connection_id
        }
    else:
        return {"error": f"Connection {connection_id} not found"}

@router.get("/connections")
async def get_active_connections():
    connections = []
    for conn_id in websocket_manager.active_connections.keys():
        if websocket_manager.is_connected(conn_id):
            buffer_content = websocket_manager.get_buffer_content(conn_id)
            connections.append({
                "connection_id": conn_id,
                "buffer_content": buffer_content,
                "buffer_length": len(buffer_content),
                "status": "connected"
            })
    return {
        "active_connections": len(connections),
        "connections": connections
    }