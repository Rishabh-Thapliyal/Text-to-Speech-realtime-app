import { UIManager } from './ui_manager.js';
import { WebSocketManager } from './websocket_manager.js';
import { AudioManager } from './audio_manager.js';
import { CaptionManager } from './caption_manager.js';

class TTSWebSocketClient {
    constructor() {
        // Initialize UI manager
        this.ui = new UIManager();

        // Bind methods for passing as callbacks
        this.handleWebSocketOpen = this.handleWebSocketOpen.bind(this);
        this.handleWebSocketMessage = this.handleWebSocketMessage.bind(this);
        this.handleWebSocketClose = this.handleWebSocketClose.bind(this);
        this.handleWebSocketError = this.handleWebSocketError.bind(this);

        // Initialize WebSocket manager
        this.ws = new WebSocketManager({
            log: this.ui.log.bind(this.ui),
            onOpen: this.handleWebSocketOpen,
            onMessage: this.handleWebSocketMessage,
            onClose: this.handleWebSocketClose,
            onError: this.handleWebSocketError
        });

        // Initialize Audio manager
        this.audio = new AudioManager({
            audioContext: null,
            audioStatus: this.ui.audioStatus,
            chunkCount: this.ui.chunkCount,
            totalAudio: this.ui.totalAudio,
            currentDuration: this.ui.currentDuration,
            log: this.ui.log.bind(this.ui),
            playPauseBtn: this.ui.playPauseBtn
        });

        // Initialize Caption manager
        this.captions = new CaptionManager({
            captionsElement: this.ui.captions,
            log: this.ui.log.bind(this.ui),
            updateCaptionSummary: () => {} // Implement if needed
        });

        // Bind UI events
        this.bindUIEvents();
    }

    bindUIEvents() {
        if (this.ui.connectBtn) {
            this.ui.connectBtn.addEventListener('click', () => {
                const url = this.ui.wsUrl ? this.ui.wsUrl.value.trim() : '';
                this.ws.connect(url);
            });
        }
        if (this.ui.disconnectBtn) {
            this.ui.disconnectBtn.addEventListener('click', () => {
                this.ws.disconnect();
            });
        }
        if (this.ui.clearAudioBtn) {
            this.ui.clearAudioBtn.addEventListener('click', () => {
                this.audio.clearAllAudio();
            });
        }
        if (this.ui.clearCaptionsBtn) {
            this.ui.clearCaptionsBtn.addEventListener('click', () => {
                this.captions.clearAllCaptions();
            });
        }
        if (this.ui.stopBtn) {
            this.ui.stopBtn.addEventListener('click', () => {
                this.audio.stopAllAudio();
            });
        }
        if (this.ui.playPauseBtn) {
            this.ui.playPauseBtn.addEventListener('click', () => {
                this.togglePlayPause();
            });
        }
        // Stream Text button event
        if (this.ui.streamBtn) {
            this.ui.streamBtn.addEventListener('click', () => {
                this.streamText();
            });
        }
        // Flush button event
        if (this.ui.flushBtn) {
            this.ui.flushBtn.addEventListener('click', () => {
                this.flushAudio();
            });
        }
        // Add more UI event bindings as needed

        // Switch Model button event
        if (this.ui.switchModelBtn) {
            this.ui.switchModelBtn.addEventListener('click', async () => {
                await this.handleSwitchModel();
            });
        }

        // Refresh current model status
        if (this.ui.refreshModelBtn) {
            this.ui.refreshModelBtn.addEventListener('click', async () => {
                await this.refreshModelStatus();
            });
        }

        // Refresh server config (and possibly reinit model)
        if (this.ui.refreshConfigBtn) {
            this.ui.refreshConfigBtn.addEventListener('click', async () => {
                await this.refreshServerConfiguration();
            });
        }
    }

    async streamText() {
        const text = this.ui.textInput ? this.ui.textInput.value.trim() : '';
        if (!text) {
            this.ui.log('Please enter text to stream.', 'warning');
            return;
        }

        try {
            // Step 1: Send initial space character (first chunk)
            this.ui.log('Sending initial space character...', 'info');
            this.ws.send(JSON.stringify({ text: " ", flush: false }));
            
            // Step 2: Send actual text
            this.ui.log(`Streaming text: "${text}"`, 'info');
            this.ws.send(JSON.stringify({ text: text, flush: false }));
            
            // Step 3: Force audio generation with flush
            this.ui.log('Forcing audio generation...', 'info');
            this.ws.send(JSON.stringify({ text: "", flush: true }));
            
            this.ui.log('Text streaming completed successfully.', 'success');
        } catch (error) {
            this.ui.log('Failed to stream text: ' + error.message, 'error');
        }
    }

    flushAudio() {
        this.ui.log('Flushing audio buffer...', 'info');
        this.ws.send(JSON.stringify({ text: "", flush: true }));
    }

    togglePlayPause() {
        // Call the AudioManager's togglePlayPause method
        this.audio.togglePlayPause();
    }

    handleWebSocketOpen() {
        this.ui.log('WebSocket connection established.', 'success');
        if (this.ui.connectionStatus) {
            this.ui.connectionStatus.textContent = 'Connected';
            this.ui.connectionStatus.className = 'connection-status status-connected';
        }
        // Enable Stream Text button
        if (this.ui.streamBtn) {
            this.ui.streamBtn.disabled = false;
        }
        // Enable Flush button
        if (this.ui.flushBtn) {
            this.ui.flushBtn.disabled = false;
        }
        // Enable Switch Model button
        if (this.ui.switchModelBtn) {
            this.ui.switchModelBtn.disabled = false;
        }
        // Update model status
        this.refreshModelStatus();
        // Enable model selection if needed
        if (this.ui.modelSelect) {
            this.ui.modelSelect.disabled = false;
        }
    }

    handleWebSocketClose() {
        this.ui.log('WebSocket connection closed.', 'warning');
        if (this.ui.connectionStatus) {
            this.ui.connectionStatus.textContent = 'Disconnected';
            this.ui.connectionStatus.className = 'connection-status status-disconnected';
        }
        this.audio.stopAllAudio();

        // Disable controls that require connection
        if (this.ui.switchModelBtn) this.ui.switchModelBtn.disabled = true;
        if (this.ui.streamBtn) this.ui.streamBtn.disabled = true;
        if (this.ui.flushBtn) this.ui.flushBtn.disabled = true;
    }

    handleWebSocketError(event) {
        this.ui.log('WebSocket encountered an error.', 'error');
        if (this.ui.connectionStatus) {
            this.ui.connectionStatus.textContent = 'Error';
            this.ui.connectionStatus.className = 'connection-status status-disconnected';
        }
    }

    async handleWebSocketMessage(data) {
        let message;
        try {
            message = JSON.parse(data);
            this.ui.log(`Received WebSocket message: ${JSON.stringify(message).substring(0, 200)}...`, 'info');
        } catch (e) {
            this.ui.log('Failed to parse WebSocket message: ' + e.message, 'error');
            return;
        }

        if (message.error) {
            this.ui.log('Server error: ' + message.error, 'error');
            return;
        }

        if (message.audio && message.alignment) {
            this.ui.log(`Processing audio response: audio length=${message.audio.length}, alignment chars=${message.alignment.chars?.length || 0}`, 'info');
            
            // Update captions
            try {
                this.captions.updateCaptions(message.alignment, message.audio);
                this.ui.log('Captions updated successfully', 'info');
            } catch (error) {
                this.ui.log('Failed to update captions: ' + error.message, 'error');
            }

            // Play audio
            try {
                this.ui.log('Processing audio chunk...', 'info');
                await this.audio.processAudioChunk(message.audio, message.alignment);
                this.ui.log('Audio chunk processed successfully', 'success');
            } catch (error) {
                this.ui.log('Failed to process audio chunk: ' + error.message, 'error');
            }
        } else if (message.status) {
            this.ui.log('Server status: ' + message.status, 'info');
        } else {
            this.ui.log('Received message without audio/alignment: ' + JSON.stringify(message), 'warning');
        }
    }

    getApiBase() {
        try {
            const urlValue = this.ui.wsUrl ? this.ui.wsUrl.value.trim() : '';
            if (urlValue) {
                const wsUrl = new URL(urlValue);
                const protocol = wsUrl.protocol === 'wss:' ? 'https:' : 'http:';
                return `${protocol}//${wsUrl.host}`;
            }
        } catch (e) {
            // ignore, fall back to location
        }
        try {
            if (window && window.location && window.location.origin) {
                return window.location.origin;
            }
        } catch (e) {}
        // Sensible default
        return 'http://localhost:8001';
    }

    updateModelStatusUI(modelType) {
        if (!this.ui.modelStatus) return;
        const type = (modelType || '').toLowerCase();
        if (type === 'kokoro') {
            this.ui.modelStatus.textContent = 'Kokoro Active';
            this.ui.modelStatus.className = 'model-status status-kokoro';
        } else if (type === 'chatterbox') {
            this.ui.modelStatus.textContent = 'Chatterbox Active';
            this.ui.modelStatus.className = 'model-status status-chatterbox';
        } else {
            this.ui.modelStatus.textContent = 'Unknown';
            this.ui.modelStatus.className = 'model-status status-unknown';
        }
    }

    async handleSwitchModel() {
        try {
            const selectedModel = this.ui.modelSelect ? this.ui.modelSelect.value : '';
            if (!selectedModel) {
                this.ui.log('Please select a model to switch.', 'warning');
                return;
            }
            const apiBase = this.getApiBase();
            this.ui.log(`Switching model to: ${selectedModel}`, 'info');
            const resp = await fetch(`${apiBase}/models/switch/${selectedModel}`, { method: 'POST' });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                this.ui.log(`Failed to switch model: ${data.error || resp.statusText}`, 'error');
                return;
            }
            this.ui.log(data.message || 'Model switched.', 'success');
            const current = data.current_model || {};
            this.updateModelStatusUI(current.model_type || selectedModel);
        } catch (error) {
            this.ui.log('Error switching model: ' + error.message, 'error');
        }
    }

    async refreshModelStatus() {
        try {
            const apiBase = this.getApiBase();
            const resp = await fetch(`${apiBase}/models/current`, { method: 'GET' });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                this.ui.log('Failed to get current model: ' + (data.error || resp.statusText), 'error');
                return;
            }
            this.updateModelStatusUI(data.model_type);
        } catch (error) {
            this.ui.log('Error fetching model status: ' + error.message, 'error');
        }
    }

    async refreshServerConfiguration() {
        try {
            const apiBase = this.getApiBase();
            const resp = await fetch(`${apiBase}/models/refresh`, { method: 'POST' });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                this.ui.log('Failed to refresh configuration: ' + (data.error || resp.statusText), 'error');
                return;
            }
            this.ui.log(data.message || 'Configuration refreshed.', 'success');
            const current = data.current_model || {};
            this.updateModelStatusUI(current.model_type);
        } catch (error) {
            this.ui.log('Error refreshing configuration: ' + error.message, 'error');
        }
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.ttsClient = new TTSWebSocketClient();
});