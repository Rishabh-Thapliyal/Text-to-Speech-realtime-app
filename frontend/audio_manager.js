export class AudioManager {
    constructor({ audioContext, audioStatus, chunkCount, totalAudio, currentDuration, log }) {
        this.audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
        this.audioStatus = audioStatus;
        this.chunkCount = chunkCount;
        this.totalAudio = totalAudio;
        this.currentDuration = currentDuration;
        this.log = log;

        this.audioQueue = [];
        this.currentAudioSource = null;
        this.currentAudioItem = null;
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
            if (this.currentDuration) this.currentDuration.textContent = durationMs;

            this.audioQueue.push({
                buffer: audioBuffer,
                alignment: alignment,
                timestamp: Date.now()
            });

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

    async createAudioBufferFromPCM(pcmData) {
        const sampleRate = 44100;
        const channels = 1;
        const bytesPerSample = 2;
        const sampleCount = pcmData.length / bytesPerSample;
        const audioBuffer = this.audioContext.createBuffer(channels, sampleCount, sampleRate);
        const channelData = audioBuffer.getChannelData(0);

        // Little-endian PCM 16-bit
        for (let i = 0; i < sampleCount; i++) {
            const sampleIndex = i * bytesPerSample;
            const sample = (pcmData[sampleIndex + 1] << 8) | pcmData[sampleIndex];
            channelData[i] = sample / 32768.0;
        }
        return audioBuffer;
    }

    playNextAudio(onAudioEnd) {
        if (this.audioQueue.length === 0) {
            this.currentAudioSource = null;
            if (this.audioStatus) this.audioStatus.textContent = 'Ready';
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
            this.playNextAudio(onAudioEnd);
        };

        this.currentAudioSource.connect(this.audioContext.destination);
        this.currentAudioSource.start();

        if (this.audioStatus) this.audioStatus.textContent = 'Playing';
        this.log(`Playing audio chunk (${Math.round(audioItem.buffer.duration * 1000)}ms)`, 'info');
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
    }

    clearAllAudio() {
        this.stopAllAudio();
        this.audioQueue = [];
        if (this.chunkCount) this.chunkCount.textContent = '0';
        if (this.totalAudio) this.totalAudio.textContent = '0';
        if (this.currentDuration) this.currentDuration.textContent = '0';
        if (this.audioStatus) this.audioStatus.textContent = 'Ready';
        this.log('Cleared all audio data', 'info');
        }
}