export class AudioManager {
    constructor({ audioContext, audioStatus, chunkCount, totalAudio, currentDuration, log, playPauseBtn }) {
        this.audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
        this.audioStatus = audioStatus;
        this.chunkCount = chunkCount;
        this.totalAudio = totalAudio;
        this.currentDuration = currentDuration;
        this.log = log;
        this.playPauseBtn = playPauseBtn;
        
        // Debug: Log which UI elements are available
        console.log('AudioManager UI Elements:', {
            audioStatus: !!this.audioStatus,
            chunkCount: !!this.chunkCount,
            totalAudio: !!this.totalAudio,
            currentDuration: !!this.currentDuration,
            playPauseBtn: !!this.playPauseBtn
        });

        this.audioQueue = [];
        this.currentAudioSource = null;
        this.currentAudioItem = null;
        
        // Track statistics
        this.totalChunks = 0;
        this.totalBytes = 0;
        this.currentChunkDuration = 0;
        
        // Debug UI testing removed for production
    }
    
    testUIUpdate() {
        this.log('Testing UI update with dummy data...', 'info');
        this.totalChunks = 1;
        this.totalBytes = 1024;
        this.currentChunkDuration = 500;
        this.updateUI();
        
        // Reset after test
        setTimeout(() => {
            this.totalChunks = 0;
            this.totalBytes = 0;
            this.currentChunkDuration = 0;
            this.updateUI();
            this.log('UI test completed, reset to zero', 'info');
        }, 2000);
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async processAudioChunk(audioBase64, alignment, onAudioEnd) {
        try {
            this.log(`Processing audio chunk: ${audioBase64.length} Base64 characters`, 'info');
            const audioData = Uint8Array.from(atob(audioBase64), c => c.charCodeAt(0));
            this.log(`Decoded to ${audioData.length} PCM bytes`, 'info');

            if (audioData.length === 0) throw new Error('Empty PCM data received');
            if (audioData.length % 2 !== 0) throw new Error(`Invalid PCM data length: ${audioData.length} bytes (must be even for 16-bit samples)`);

            const audioBuffer = await this.createAudioBufferFromPCM(audioData);
            this.log(`Created AudioBuffer: ${audioBuffer.length} samples, ${audioBuffer.duration.toFixed(3)}s`, 'info');

            const durationMs = Math.round(audioBuffer.duration * 1000);
            
            // Update statistics IMMEDIATELY
            this.log(`About to update statistics...`, 'info');
            this.totalChunks++;
            this.totalBytes += audioData.length;
            this.currentChunkDuration = durationMs;
            this.log(`Statistics updated - Chunks: ${this.totalChunks}, Bytes: ${this.totalBytes}, Duration: ${this.currentChunkDuration}ms`, 'info');
            
            // Enqueue audio, then update UI so controls reflect availability
            this.audioQueue.push({
                buffer: audioBuffer,
                alignment: alignment,
                timestamp: Date.now()
            });

            // Update UI elements after queueing so play button enables correctly
            this.log(`About to update UI...`, 'info');
            this.updateUI();
            this.log(`UI update completed`, 'info');

            if (!this.currentAudioSource) {
                this.playNextAudio(onAudioEnd);
            }

            this.log(`Audio chunk processed successfully: ${durationMs}ms duration`, 'success');
        } catch (error) {
            this.log('Failed to process audio chunk: ' + error.message, 'error');
            this.log('Audio data length: ' + (audioBase64 ? audioBase64.length : 'undefined'), 'error');
            this.log('Alignment data: ' + JSON.stringify(alignment), 'error');
        }
    }

    updateUI() {
        this.log(`Updating UI - Chunks: ${this.totalChunks}, Bytes: ${this.totalBytes}, Duration: ${this.currentChunkDuration}ms`, 'info');
        
        // Assume DOM references are provided by constructor

        // Update chunk count
        if (this.chunkCount) {
            this.chunkCount.textContent = this.totalChunks.toString();
            this.log(`Updated chunk count to: ${this.totalChunks} (DOM shows: "${this.chunkCount.textContent}")`, 'info');
        } else {
            this.log('ERROR: chunkCount element not found!', 'error');
        }
        
        // Update total audio bytes (formatted)
        if (this.totalAudio) {
            const formattedBytes = this.formatBytes(this.totalBytes);
            this.totalAudio.textContent = formattedBytes;
            this.log(`Updated total audio to: ${formattedBytes} (DOM shows: "${this.totalAudio.textContent}")`, 'info');
        } else {
            this.log('ERROR: totalAudio element not found!', 'error');
        }
        
        // Update current duration
        if (this.currentDuration) {
            this.currentDuration.textContent = this.currentChunkDuration.toString();
            this.log(`Updated current duration to: ${this.currentChunkDuration}ms (DOM shows: "${this.currentDuration.textContent}")`, 'info');
        } else {
            this.log('ERROR: currentDuration element not found!', 'error');
        }

        // Enable/disable play button based on audio availability or active playback
        if (this.playPauseBtn) {
            this.playPauseBtn.disabled = (this.audioQueue.length === 0 && !this.currentAudioSource);
            this.log(`Play button ${this.playPauseBtn.disabled ? 'disabled' : 'enabled'} (queue length: ${this.audioQueue.length})`, 'info');
        } else {
            this.log('Warning: playPauseBtn element not found', 'warning');
        }
    }

    async createAudioBufferFromPCM(pcmData) {
        const sampleRate = 44100;
        const channels = 1;
        const bytesPerSample = 2;
        const sampleCount = pcmData.length / bytesPerSample;
        const audioBuffer = this.audioContext.createBuffer(channels, sampleCount, sampleRate);
        const channelData = audioBuffer.getChannelData(0);

        // Little-endian PCM 16-bit, signed two's complement → float [-1, 1)
        for (let i = 0; i < sampleCount; i++) {
            const sampleIndex = i * bytesPerSample;
            let int16 = (pcmData[sampleIndex + 1] << 8) | pcmData[sampleIndex];
            if (int16 & 0x8000) int16 -= 0x10000; // sign extend
            channelData[i] = int16 / 32768.0;
        }
        return audioBuffer;
    }

    togglePlayPause() {
        if (this.currentAudioSource && this.currentAudioSource.playbackState === 'playing') {
            // Pause current audio
            this.currentAudioSource.stop();
            this.currentAudioSource = null;
            this.currentAudioItem = null;
            if (this.audioStatus) this.audioStatus.textContent = 'Paused';
            if (this.playPauseBtn) this.playPauseBtn.textContent = '▶️ Play All';
            this.log('Audio playback paused', 'info');
        } else {
            // Resume or start playing
            if (this.audioQueue.length > 0 || this.currentAudioItem) {
                if (this.currentAudioItem) {
                    // Resume current audio
                    this.playNextAudio();
                } else {
                    // Start playing from queue
                    this.playNextAudio();
                }
                if (this.playPauseBtn) this.playPauseBtn.textContent = '⏸️ Pause';
            }
        }
    }

    playNextAudio(onAudioEnd) {
        if (this.audioQueue.length === 0) {
            this.currentAudioSource = null;
            if (this.audioStatus) this.audioStatus.textContent = 'Ready';
            if (this.playPauseBtn) this.playPauseBtn.textContent = '▶️ Play All';
            return;
        }

        const audioItem = this.audioQueue.shift();
        this.currentAudioSource = this.audioContext.createBufferSource();
        this.currentAudioSource.buffer = audioItem.buffer;
        this.currentAudioItem = audioItem;

        this.currentAudioSource.onended = () => {
            this.currentAudioSource = null;
            this.currentAudioItem = null;
            if (typeof onAudioEnd === 'function') onAudioEnd(audioItem);
            // Update UI after a chunk finishes playing
            this.updateUI();
            this.playNextAudio(onAudioEnd);
        };

        this.currentAudioSource.connect(this.audioContext.destination);
        this.currentAudioSource.start();

        if (this.audioStatus) this.audioStatus.textContent = 'Playing';
        if (this.playPauseBtn) this.playPauseBtn.textContent = '⏸️ Pause';
        this.log(`Playing audio chunk (${Math.round(audioItem.buffer.duration * 1000)}ms)`, 'info');
        // Ensure UI reflects active playback state immediately
        this.updateUI();
    }

    stopAllAudio() {
        if (this.currentAudioSource) {
            this.currentAudioSource.stop();
            this.currentAudioSource = null;
            this.currentAudioItem = null;
            if (this.audioStatus) this.audioStatus.textContent = 'Stopped';
            this.log('Stopped all audio playback', 'info');
        }
        this.audioQueue = [];
        // Reflect stopped state in UI
        this.updateUI();
    }

    clearAllAudio() {
        this.stopAllAudio();
        this.audioQueue = [];
        
        // Reset statistics
        this.totalChunks = 0;
        this.totalBytes = 0;
        this.currentChunkDuration = 0;
        
        // Update UI
        this.updateUI();
        
        // Reset button state
        if (this.playPauseBtn) {
            this.playPauseBtn.textContent = '▶️ Play All';
            this.playPauseBtn.disabled = true;
        }
        
        if (this.audioStatus) this.audioStatus.textContent = 'Ready';
        this.log('Cleared all audio data', 'info');
    }
}