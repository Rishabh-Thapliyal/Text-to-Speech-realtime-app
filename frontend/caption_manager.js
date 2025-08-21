export class CaptionManager {
    constructor({ captionsElement, log, updateCaptionSummary }) {
        this.captions = captionsElement;
        this.log = log;
        this.updateCaptionSummary = updateCaptionSummary;
        this.alignmentData = [];
        this.chunkCounter = 0;
        this.characterHighlightInterval = null;
    }

    findExistingChunkForAlignment(alignment) {
        if (!this.alignmentData || this.alignmentData.length === 0) {
            return null;
        }
        
        // For real-time streaming, we want to allow new chunks even if they have similar content
        // Only prevent exact duplicates with identical text and very similar timing
        for (const data of this.alignmentData) {
            if (this.areAlignmentsEqual(data.alignment, alignment)) {
                return data.chunkId;
            }
        }
        return null;
    }

    areAlignmentsEqual(alignment1, alignment2) {
        if (!alignment1 || !alignment2) return false;
        
        // Check if text content is exactly the same
        if (alignment1.chars.join('') !== alignment2.chars.join('')) {
            return false;
        }
        
        // Check if arrays have same length
        if (alignment1.chars.length !== alignment2.chars.length) {
            return false;
        }
        
        // For real-time streaming, allow some timing variation
        // Only consider it a duplicate if timing is very similar (within 5ms)
        const timingTolerance = 5; // 5ms tolerance for real-time streaming
        
        for (let i = 0; i < alignment1.char_start_times_ms.length; i++) {
            if (Math.abs(alignment1.char_start_times_ms[i] - alignment2.char_start_times_ms[i]) > timingTolerance) {
                return false;
            }
        }
        
        for (let i = 0; i < alignment1.char_durations_ms.length; i++) {
            if (Math.abs(alignment1.char_durations_ms[i] - alignment2.char_durations_ms[i]) > timingTolerance) {
                return false;
            }
        }
        
        return true;
    }

    getNextChunkNumber() {
        if (!this.chunkCounter) {
            this.chunkCounter = 0;
        }
        return ++this.chunkCounter;
    }

    validateAlignmentFormat(alignment) {
        if (!alignment.chars || !Array.isArray(alignment.chars) ||
            !alignment.char_start_times_ms || !Array.isArray(alignment.char_start_times_ms) ||
            !alignment.char_durations_ms || !Array.isArray(alignment.char_durations_ms)) {
            this.log('⚠️ Basic alignment validation failed', 'warning');
            return false;
        }
        if (alignment.chars.length !== alignment.char_start_times_ms.length ||
            alignment.chars.length !== alignment.char_durations_ms.length) {
            this.log('⚠️ Alignment array length mismatch', 'warning');
            return false;
        }
        this.log(`✅ Alignment validation passed: ${alignment.chars.length} characters`, 'success');
        return true;
    }

    highlightCharacter(chunkId, charIndex, state) {
        if (!chunkId || charIndex === undefined) {
            this.log('Warning: Invalid parameters for character highlighting', 'warning');
            return;
        }
        const chunk = document.querySelector(`[data-chunk-id="${chunkId}"]`);
        if (!chunk) {
            this.log(`Warning: Chunk ${chunkId} not found for highlighting`, 'warning');
            return;
        }
        const charSpan = chunk.querySelector(`[data-index="${charIndex}"]`);
        if (!charSpan) {
            this.log(`Warning: Character ${charIndex} not found in chunk ${chunkId}`, 'warning');
            return;
        }
        charSpan.classList.remove('active', 'playing', 'played');
        if (state === 'active') {
            charSpan.classList.add('active');
        } else if (state === 'playing') {
            charSpan.classList.add('playing');
            charSpan.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else if (state === 'played') {
            charSpan.classList.add('played');
        }
    }

    clearChunkHighlighting(chunkId) {
        if (!chunkId) {
            this.log('Warning: Invalid chunk ID for clearing highlighting', 'warning');
            return;
        }
        const chunk = document.querySelector(`[data-chunk-id="${chunkId}"]`);
        if (!chunk) {
            this.log(`Warning: Chunk ${chunkId} not found for clearing highlighting`, 'warning');
            return;
        }
        chunk.querySelectorAll('.caption-char').forEach(charSpan => {
            charSpan.classList.remove('active', 'playing', 'played');
        });
    }

    updateCaptions(alignment, audioBase64 = null) {
        if (!alignment.chars || alignment.chars.length === 0) {
            this.log('⚠️ Empty alignment data received, skipping caption update', 'warning');
            return;
        }
        
        this.log(`🔄 Processing caption chunk: "${alignment.chars.join('')}" (${alignment.chars.length} chars)`, 'info');
        if (alignment.chars.length > 20) {
            this.log(`⚠️ Warning: Chunk contains ${alignment.chars.length} characters - this might be the entire text instead of a chunk`, 'warning');
        }
        
        // Add timestamp to make each chunk unique for real-time streaming
        const timestamp = Date.now();
        const uniqueAlignment = {
            ...alignment,
            timestamp: timestamp,
            chunkId: `chunk_${timestamp}_${Math.random().toString(36).substr(2, 9)}`
        };
        
        this.log(`🔍 Checking for duplicate chunks...`, 'debug');
        const existingChunk = this.findExistingChunkForAlignment(uniqueAlignment);
        if (existingChunk) {
            this.log(`⚠️ Chunk already exists for this alignment, skipping duplicate (ID: ${existingChunk})`, 'info');
            return;
        }
        
        this.log(`✅ No duplicates found, creating new caption chunk...`, 'debug');
        const isValid = this.validateAlignmentFormat(alignment);
        if (!isValid) {
            this.log('⚠️ Displaying captions despite validation errors', 'warning');
        }
        
        const captionContainer = document.createElement('div');
        captionContainer.className = 'caption-chunk';
        const chunkId = uniqueAlignment.chunkId;
        captionContainer.dataset.chunkId = chunkId;
        captionContainer.style.textAlign = 'center';
        captionContainer.style.padding = '15px';
        captionContainer.style.margin = '10px 0';
        captionContainer.style.fontSize = '18px';
        captionContainer.style.lineHeight = '1.6';
        captionContainer.style.border = '1px solid #e0e0e0';
        captionContainer.style.borderRadius = '8px';
        captionContainer.style.backgroundColor = '#fafafa';
        
        const chunkNumber = this.getNextChunkNumber();
        const header = document.createElement('div');
        header.style.marginBottom = '15px';
        header.style.fontSize = '14px';
        header.style.color = '#666';
        header.style.borderBottom = '1px solid #e0e0e0';
        header.style.paddingBottom = '10px';
        const debugInfo = `[Debug: Text="${alignment.chars.join('')}", Chunk#=${chunkNumber}, Time=${timestamp}]`;
        header.innerHTML = `<strong>Chunk ${chunkNumber}:</strong> ${alignment.chars.length} characters, ${Math.max(...alignment.char_durations_ms)}ms max duration <small style="color: #999;">${debugInfo}</small>`;
        captionContainer.appendChild(header);
        
        alignment.chars.forEach((char, index) => {
            const charSpan = document.createElement('span');
            charSpan.className = 'caption-char';
            charSpan.textContent = char;
            charSpan.dataset.index = index;
            charSpan.dataset.startTime = alignment.char_start_times_ms[index];
            charSpan.dataset.duration = alignment.char_durations_ms[index];
            charSpan.dataset.chunkNumber = chunkNumber;
            charSpan.dataset.timestamp = timestamp;
            const startTime = alignment.char_start_times_ms[index];
            const duration = alignment.char_durations_ms[index];
            charSpan.title = `Chunk ${chunkNumber}, Start: ${startTime}ms, Duration: ${duration}ms, Time: ${timestamp}`;
            captionContainer.appendChild(charSpan);
        });
        
        this.captions.appendChild(captionContainer);
        if (!this.alignmentData) this.alignmentData = [];
        this.alignmentData.push({
            chunkId: chunkId,
            alignment: uniqueAlignment,
            audioBase64: audioBase64,
            timestamp: timestamp
        });
        
        this.displayAlignmentData(uniqueAlignment, chunkNumber, chunkId);
        this.updateCaptionSummary();
        this.log(`✅ Caption chunk ${chunkNumber} added successfully: "${alignment.chars.join('')}" (${alignment.chars.length} characters, ID: ${chunkId})`, 'success');
    }

    displayAlignmentData(alignment, chunkNumber, chunkId) {
        if (!alignment.chars || alignment.chars.length === 0) return;
        const alignmentDisplay = document.createElement('div');
        alignmentDisplay.style.marginTop = '15px';
        alignmentDisplay.style.padding = '15px';
        alignmentDisplay.style.backgroundColor = '#f8f9fa';
        alignmentDisplay.style.borderRadius = '8px';
        alignmentDisplay.style.fontFamily = 'monospace';
        alignmentDisplay.style.fontSize = '12px';
        alignmentDisplay.style.textAlign = 'left';
        alignmentDisplay.style.overflowX = 'auto';
        alignmentDisplay.style.border = '1px solid #dee2e6';
        const formattedData = {
            chars: alignment.chars,
            char_start_times_ms: alignment.char_start_times_ms,
            char_durations_ms: alignment.char_durations_ms
        };
        alignmentDisplay.innerHTML = `
            <h5 style="margin-bottom: 10px; color: #495057;">📊 Chunk ${chunkNumber} - Character Alignment Data (Raw Format)</h5>
            <p style="margin-bottom: 10px; color: #666; font-size: 11px;">
                <strong>Chunk Text:</strong> "${alignment.chars.join('')}"<br>
                <strong>Characters:</strong> ${alignment.chars.length}<br>
                <strong>Chunk ID:</strong> ${chunkId}
            </p>
            <pre style="margin: 0; white-space: pre-wrap;">${JSON.stringify(formattedData, null, 2)}</pre>
        `;
        const currentChunk = document.querySelector(`[data-chunk-id="${chunkId}"]`);
        if (currentChunk) {
            currentChunk.appendChild(alignmentDisplay);
        } else {
            this.log(`Warning: Could not find chunk with ID ${chunkId} for alignment display`, 'warning');
        }
    }

    clearAllCaptions() {
        if (!this.captions) {
            this.log('Error: Captions element not found', 'error');
            return;
        }
        this.captions.innerHTML = `
            <div class="empty-state">
                <p>🎬 Captions will appear here as audio is generated...</p>
                <small>Characters will highlight in real-time based on audio playback</small>
            </div>
        `;
        this.updateCaptionSummary();
        this.alignmentData = [];
        this.chunkCounter = 0;
        this.log('All caption chunks cleared', 'info');
    }

    clearChunkById(chunkId) {
        if (!chunkId) return;
        const chunk = document.querySelector(`[data-chunk-id="${chunkId}"]`);
        if (chunk) {
            chunk.remove();
            this.log(`Chunk ${chunkId} cleared`, 'info');
            if (this.alignmentData) {
                this.alignmentData = this.alignmentData.filter(data => data.chunkId !== chunkId);
            }
            this.updateCaptionSummary();
        }
    }
    
    startNewStreamingSession() {
        // Clear all existing captions when starting a new streaming session
        this.clearAllCaptions();
        this.log('🎬 New streaming session started - captions cleared', 'info');
    }
    
    updateCaptionSummary() {
        if (!this.updateCaptionSummary) return;
        
        try {
            const totalChunks = this.alignmentData ? this.alignmentData.length : 0;
            const totalCharacters = this.alignmentData ? 
                this.alignmentData.reduce((sum, data) => sum + (data.alignment.chars ? data.alignment.chars.length : 0), 0) : 0;
            
            const summaryElement = document.getElementById('captionSummary');
            if (summaryElement) {
                if (totalChunks === 0) {
                    summaryElement.innerHTML = '<span>No captions yet</span>';
                } else {
                    summaryElement.innerHTML = `<span>${totalChunks} chunks, ${totalCharacters} characters</span>`;
                }
            }
        } catch (error) {
            this.log(`Error updating caption summary: ${error.message}`, 'warning');
        }
    }
}