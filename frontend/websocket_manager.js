export class WebSocketManager {
    constructor({ log, onOpen, onMessage, onClose, onError }) {
        this.websocket = null;
        this.isConnected = false;
        this.log = log;
        this.onOpen = onOpen;
        this.onMessage = onMessage;
        this.onClose = onClose;
        this.onError = onError;
    }

    connect(url) {
        if (!url) {
            this.log('Please enter a WebSocket URL', 'error');
            return;
        }
        try {
            this.log('Attempting to connect to: ' + url, 'info');
            this.websocket = new WebSocket(url);

            this.websocket.onopen = (event) => {
                this.isConnected = true;
                this.log('WebSocket connected successfully', 'success');
                if (typeof this.onOpen === 'function') this.onOpen(event);
            };

            this.websocket.onmessage = (event) => {
                if (typeof this.onMessage === 'function') this.onMessage(event.data);
            };

            this.websocket.onclose = (event) => {
                this.isConnected = false;
                this.log('WebSocket disconnected', 'warning');
                if (typeof this.onClose === 'function') this.onClose(event);
            };

            this.websocket.onerror = (event) => {
                this.log('WebSocket error: ' + (event.message || 'Connection failed'), 'error');
                if (typeof this.onError === 'function') this.onError(event);
            };

            this.log('WebSocket object created, waiting for connection...', 'info');
        } catch (error) {
            this.log('Failed to create WebSocket: ' + error.message, 'error');
            console.error('WebSocket creation error:', error);
        }
    }

    send(data) {
        if (this.websocket && this.isConnected && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(data);
        } else {
            this.log('WebSocket is not open. Cannot send message.', 'error');
        }
    }

    disconnect() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        this.isConnected = false;
    }
}