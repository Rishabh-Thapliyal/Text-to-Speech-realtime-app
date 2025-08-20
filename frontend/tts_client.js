import { UIManager } from './ui_manager_2.js';
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
            log: this.ui.log.bind(this.ui)
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
        // Stream Text button event
        if (this.ui.streamBtn) {
            this.ui.streamBtn.addEventListener('click', () => {
                const text = this.ui.textInput ? this.ui.textInput.value.trim() : '';
                if (!text) {
                    this.ui.log('Please enter text to stream.', 'warning');
                    return;
                }
                // Send text to server (can be customized for chunking/spec)
                this.ws.send(JSON.stringify({ text }));
                this.ui.log('Text sent for streaming.', 'info');
            });
        }
        // Add more UI event bindings as needed
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
        // Update model status
        if (this.ui.modelStatus) {
            this.ui.modelStatus.textContent = 'Connected';
            this.ui.modelStatus.className = 'model-status status-chatterbox'; // or appropriate class
        }
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
        } catch (e) {
            this.ui.log('Failed to parse WebSocket message: ' + e.message, 'error');
            return;
        }

        if (message.error) {
            this.ui.log('Server error: ' + message.error, 'error');
            return;
        }

        if (message.audio && message.alignment) {
            // Update captions
            this.captions.updateCaptions(message.alignment, message.audio);

            // Play audio
            await this.audio.processAudioChunk(message.audio, message.alignment);
        }
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.ttsClient = new TTSWebSocketClient();
});