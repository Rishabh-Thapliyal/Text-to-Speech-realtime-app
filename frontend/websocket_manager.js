export class WebSocketManager {
    constructor({ log, onOpen, onMessage, onClose, onError, onReconnectAttempt }) {
        this.websocket = null;
        this.isConnected = false;
        this.log = log;
        this.onOpen = onOpen;
        this.onMessage = onMessage;
        this.onClose = onClose;
        this.onError = onError;
        this.onReconnectAttempt = onReconnectAttempt;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        this.reconnectTimer = null;
        this.url = null;
        this.shouldReconnect = true;
    }

    connect(url) {
        if (!url) {
            this.log('Please enter a WebSocket URL', 'error');
            return;
        }
        
        this.url = url;
        this.shouldReconnect = true;
        this._connectInternal();
    }

    _connectInternal() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.log('WebSocket already connected', 'info');
            return;
        }

        try {
            this.log('Attempting to connect to: ' + this.url, 'info');
            this.websocket = new WebSocket(this.url);

            this.websocket.onopen = (event) => {
                this.isConnected = true;
                this.reconnectAttempts = 0; // Reset reconnect attempts on successful connection
                this.reconnectDelay = 1000; // Reset delay
                this.log('WebSocket connected successfully', 'success');
                if (typeof this.onOpen === 'function') this.onOpen(event);
            };

            this.websocket.onmessage = (event) => {
                if (typeof this.onMessage === 'function') this.onMessage(event.data);
            };

            this.websocket.onclose = (event) => {
                this.isConnected = false;
                this.log('WebSocket disconnected', 'warning');
                
                // Attempt to reconnect if not manually disconnected
                if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
                    this._scheduleReconnect();
                } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                    this.log('Max reconnection attempts reached', 'error');
                }
                
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
            
            // Schedule reconnect on creation failure
            if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
                this._scheduleReconnect();
            }
        }
    }

    _scheduleReconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
        }
        
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1); // Exponential backoff
        
        this.log(`Scheduling reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`, 'info');
        
        // Notify about reconnection attempt
        if (typeof this.onReconnectAttempt === 'function') {
            this.onReconnectAttempt(this.reconnectAttempts, this.maxReconnectAttempts);
        }
        
        this.reconnectTimer = setTimeout(() => {
            this.log(`Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts}`, 'info');
            this._connectInternal();
        }, delay);
    }

    send(data) {
        if (this.websocket && this.isConnected && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(data);
        } else {
            this.log('WebSocket is not open. Cannot send message.', 'error');
        }
    }

    disconnect() {
        this.shouldReconnect = false; // Prevent automatic reconnection
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        this.isConnected = false;
    }

    // Method to manually trigger reconnection
    reconnect() {
        this.shouldReconnect = true;
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        if (this.url) {
            this._connectInternal();
        }
    }
}