export class UIManager {
    constructor() {
        // DOM element references
        this.connectBtn = document.getElementById('connectBtn');
        this.disconnectBtn = document.getElementById('disconnectBtn');
        this.modelSelect = document.getElementById('modelSelect');
        this.modelStatus = document.getElementById('modelStatus');
        this.switchModelBtn = document.getElementById('switchModelBtn');
        this.refreshModelBtn = document.getElementById('refreshModelBtn');
        this.refreshConfigBtn = document.getElementById('refreshConfigBtn');
        this.streamBtn = document.getElementById('streamBtn');
        this.manualStreamBtn = document.getElementById('manualStreamBtn');
        this.flushBtn = document.getElementById('flushBtn');
        this.clearBufferBtn = document.getElementById('clearBufferBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.textInput = document.getElementById('textInput');
        this.wsUrl = document.getElementById('wsUrl');
        this.connectionStatus = document.getElementById('connectionStatus');
        this.audioStatus = document.getElementById('audioStatus');
        this.chunkCount = document.getElementById('chunkCount');
        this.totalAudio = document.getElementById('totalAudio');
        this.currentDuration = document.getElementById('currentDuration');
        this.captions = document.getElementById('captions');
        this.logContainer = document.getElementById('logContainer');
        this.specStatusContainer = document.getElementById('specStatusContainer');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.clearCaptionsBtn = document.getElementById('clearCaptionsBtn');
        this.playPauseBtn = document.getElementById('playPauseBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.clearAudioBtn = document.getElementById('clearAudioBtn');
        this.toggleHighlightingBtn = document.getElementById('toggleHighlightingBtn');
    }

    log(message, type = 'info') {
        // Always log to console first
        console.log(`[${type.toUpperCase()}] ${message}`);
        // Log to DOM if available
        try {
            if (this.logContainer && this.logContainer.appendChild) {
                const logEntry = document.createElement('div');
                logEntry.className = `log-entry log-${type}`;
                const timestamp = new Date().toLocaleTimeString();
                logEntry.textContent = `[${timestamp}] ${message}`;
                this.logContainer.appendChild(logEntry);
                this.logContainer.scrollTop = this.logContainer.scrollHeight;
                // Keep only last 100 log entries
                while (this.logContainer.children.length > 100) {
                    this.logContainer.removeChild(this.logContainer.firstChild);
                }
            }
        } catch (error) {
            // If DOM logging fails, just continue with console logging
            console.warn('DOM logging failed, continuing with console only:', error);
        }
    }
}