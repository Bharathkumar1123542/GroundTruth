/**
 * app.js — GroundTruth Kiosk Alpine.js Components
 * =================================================
 * All Alpine.js component factories used across the kiosk UI.
 *
 * Components:
 *   kioskShell()    — base.html: global shell state (no-op init, future: idle timeout)
 *   recorder()      — record.html: MediaRecorder, waveform, countdown, pipeline POST
 *
 * The confirmation() component is defined inline in confirmation.html because it is
 * trivial (countdown only) and has no cross-screen dependencies.
 *
 * Dependencies (loaded in base.html, served from /static/vendor/):
 *   - Alpine.js  ≥3.13   (alpine.min.js)
 *   - HTMX       ≥1.9    (htmx.min.js)
 *
 * No external fetch calls are made from this file.
 * All API calls go through HTMX declarative attributes on HTML elements,
 * keeping the JS layer thin and testable.
 */

'use strict';

/* ===========================================================================
   HTMX Global Configuration
   =========================================================================== */

/**
 * Configure HTMX before it initialises.
 * htmx.config must be set before the first htmx:load event fires.
 */
document.addEventListener('DOMContentLoaded', () => {
  if (typeof htmx !== 'undefined') {
    // Follow HX-Redirect response headers (used by Kiosk Agent to navigate
    // between screens after a successful POST /v1/complaints/record).
    htmx.config.allowEval = false;       // no eval — safe default
    htmx.config.selfRequestsOnly = true; // only talk to our own origin
    htmx.config.defaultSwapStyle = 'outerHTML';
    htmx.config.defaultSettleDelay = 120;
  }
});


/* ===========================================================================
   kioskShell() — Global shell component (x-data on <body> in base.html)
   =========================================================================== */

/**
 * kioskShell()
 *
 * Minimal shell state. Primarily exists so child components can reference
 * $root for cross-component communication if needed in future screens.
 *
 * Currently responsible for:
 *   - No-op init (placeholder for global idle-timeout in production roadmap)
 */
function kioskShell() {
  return {
    init() {
      // Future: implement global inactivity timeout that redirects to / after
      // N minutes of no interaction (production roadmap — kiosk operator concern).
    },
  };
}


/* ===========================================================================
   recorder() — Record screen component (x-data on #record-card)
   =========================================================================== */

/**
 * recorder(sessionId, language, maxDuration)
 *
 * Manages the full recording lifecycle:
 *   IDLE → RECORDING → STOPPED → PROCESSING → (redirected by HTMX) | ERROR
 *
 * Audio capture uses the Web MediaRecorder API (available in Chromium kiosk mode,
 * architecture.md §7.2). Audio is captured as WebM/Opus or OGG/Opus depending
 * on what Chromium supports, then sent as a multipart/form-data blob in the
 * finalize POST.
 *
 * @param {string} sessionId   - UUID4 session ID returned by POST /v1/complaints/record
 * @param {string} language    - ISO-639-1 language code ("hi" | "mr" | "ta")
 * @param {number} maxDuration - Max recording duration in seconds (default 90)
 */
function recorder(sessionId, language, maxDuration) {
  return {
    // ── State ──────────────────────────────────────────────────────────────
    state: 'IDLE',          // 'IDLE' | 'RECORDING' | 'PROCESSING' | 'ERROR'
    sessionId:    sessionId,
    language:     language,
    maxDuration:  maxDuration || 90,

    // Recording
    mediaRecorder:  null,   // MediaRecorder instance
    audioChunks:    [],     // Blob chunks accumulated during recording
    audioBlob:      null,   // Final Blob after recording stops
    stream:         null,   // MediaStream (held to stop tracks on cleanup)

    // Countdown
    remaining:      0,      // seconds remaining (counts down from maxDuration)
    _countdownTimer: null,

    // Waveform
    waveHeights:    Array(9).fill(8),  // array of bar heights in px (9 bars)
    _waveformTimer: null,
    _analyser:      null,   // Web Audio AnalyserNode
    _audioCtx:      null,   // AudioContext

    // Pipeline / error
    pipelineStage: 'Preparing…',
    errorMsg:      '',
    retryAllowed:  true,

    // ── Lifecycle ──────────────────────────────────────────────────────────

    init() {
      // Ensure the finalize form's action is set to the correct session URL.
      const form = document.getElementById('finalize-form');
      if (form) {
        form.setAttribute('action', `/v1/complaints/${this.sessionId}/finalize`);
        // HTMX reads hx-post at request time; keep it in sync.
        form.setAttribute('hx-post', `/v1/complaints/${this.sessionId}/finalize`);
        htmx && htmx.process(form);
      }
    },

    // ── Recording control ─────────────────────────────────────────────────

    async toggleRecording() {
      if (this.state === 'IDLE') {
        await this.startRecording();
      } else if (this.state === 'RECORDING') {
        this.stopRecording();
      }
    },

    async startRecording() {
      // Request microphone access.
      // In Chromium kiosk mode with autoGrantPermissions or a pre-granted
      // policy, this resolves immediately.
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 16000,   // match Whisper's expected input rate
            echoCancellation: true,
            noiseSuppression: true,
          },
          video: false,
        });
      } catch (err) {
        this.state = 'ERROR';
        this.errorMsg = 'Microphone access denied. Please check kiosk settings.';
        this.retryAllowed = false;
        console.error('[recorder] getUserMedia failed:', err);
        return;
      }

      // Set up Web Audio analyser for waveform visualisation.
      this._setupAnalyser(this.stream);

      // Choose a supported MIME type.
      const mimeType = this._getSupportedMime();

      this.audioChunks = [];
      try {
        this.mediaRecorder = new MediaRecorder(this.stream, { mimeType });
      } catch (err) {
        // Fallback: let the browser choose the format.
        this.mediaRecorder = new MediaRecorder(this.stream);
      }

      this.mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          this.audioChunks.push(e.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        this.audioBlob = new Blob(this.audioChunks, {
          type: this.mediaRecorder.mimeType || 'audio/webm',
        });
        this._submitFinalize();
      };

      // Collect data every 500ms so we have chunks even if stop() is abrupt.
      this.mediaRecorder.start(500);
      this.state = 'RECORDING';
      this.remaining = this.maxDuration;

      // Start countdown.
      this._countdownTimer = setInterval(() => {
        this.remaining -= 1;
        if (this.remaining <= 0) {
          this.stopRecording();  // auto-stop at max duration
        }
      }, 1000);

      // Start waveform animation.
      this._startWaveform();
    },

    stopRecording() {
      if (this.state !== 'RECORDING') return;

      // Clear countdown.
      clearInterval(this._countdownTimer);
      this._countdownTimer = null;

      // Stop waveform.
      this._stopWaveform();

      // Stop MediaRecorder — triggers onstop → _submitFinalize.
      if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
        this.mediaRecorder.stop();
      }

      // Release microphone.
      if (this.stream) {
        this.stream.getTracks().forEach(t => t.stop());
        this.stream = null;
      }

      this.state = 'PROCESSING';
      this.pipelineStage = 'Preparing audio…';
    },

    // ── Pipeline submission ────────────────────────────────────────────────

    /**
     * Appends the audio blob to the finalize form and submits it via HTMX.
     * Called from MediaRecorder.onstop after the blob is assembled.
     */
    _submitFinalize() {
      const form = document.getElementById('finalize-form');
      if (!form || !this.audioBlob) {
        this.state = 'ERROR';
        this.errorMsg = 'Recording failed — no audio captured. Please try again.';
        this.retryAllowed = true;
        return;
      }

      // Remove any previous audio_data input to avoid duplicates.
      const existing = form.querySelector('input[name="audio_data"]');
      if (existing) existing.remove();

      // Create a hidden file input carrying the audio blob.
      const dt = new DataTransfer();
      const ext = this.audioBlob.type.includes('ogg') ? 'ogg' : 'webm';
      dt.items.add(new File([this.audioBlob], `recording.${ext}`, { type: this.audioBlob.type }));

      const fileInput = document.createElement('input');
      fileInput.type = 'file';
      fileInput.name = 'audio_data';
      fileInput.files = dt.files;
      fileInput.style.display = 'none';
      form.appendChild(fileInput);

      this.pipelineStage = 'Transcribing speech…';

      // Submit via HTMX.
      if (typeof htmx !== 'undefined') {
        htmx.trigger(form, 'submit');
      } else {
        // Fallback: native form submit (no HTMX swap, but pipeline still runs).
        form.submit();
      }
    },

    /**
     * Called by HTMX @htmx:xhr:progress to update the pipeline stage label
     * as the server streams back progress (if the server implements SSE/chunked
     * responses in a future version). For MVP, we update the label on a timer.
     */
    onProgress(_event) {
      // MVP: cycle through stage labels to give the resident feedback.
      const stages = [
        'Transcribing speech…',
        'Structuring complaint…',
        'Verifying with local data…',
        'Creating your ticket…',
      ];
      let idx = stages.indexOf(this.pipelineStage);
      if (idx < stages.length - 1) {
        this.pipelineStage = stages[idx + 1];
      }
    },

    onPipelineError(event) {
      this.state = 'ERROR';
      const xhr = event.detail && event.detail.xhr;
      if (xhr) {
        try {
          const body = JSON.parse(xhr.responseText);
          this.errorMsg    = body.message      || 'Processing failed. Please try again.';
          this.retryAllowed = body.retry_allowed !== false;
        } catch {
          this.errorMsg    = 'Processing failed. Please try again.';
          this.retryAllowed = true;
        }
      } else {
        this.errorMsg    = 'Connection to the kiosk was lost. Please try again.';
        this.retryAllowed = true;
      }
    },

    resetToIdle() {
      this._cleanup();
      this.state         = 'IDLE';
      this.errorMsg      = '';
      this.retryAllowed  = true;
      this.pipelineStage = 'Preparing…';
      this.audioChunks   = [];
      this.audioBlob     = null;
      this.remaining     = 0;
      this.waveHeights   = Array(9).fill(8);
    },

    // ── Waveform helpers ──────────────────────────────────────────────────

    _getSupportedMime() {
      const candidates = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/ogg',
      ];
      for (const mime of candidates) {
        if (MediaRecorder.isTypeSupported(mime)) return mime;
      }
      return '';  // let browser decide
    },

    _setupAnalyser(stream) {
      try {
        this._audioCtx = new (window.AudioContext || window.webkitAudioContext)({
          sampleRate: 16000,
        });
        const source = this._audioCtx.createMediaStreamSource(stream);
        this._analyser = this._audioCtx.createAnalyser();
        this._analyser.fftSize = 64;
        source.connect(this._analyser);
      } catch (err) {
        // Web Audio not available — waveform will use random heights.
        console.warn('[recorder] Web Audio API unavailable:', err);
        this._analyser = null;
      }
    },

    _startWaveform() {
      const NUM_BARS  = 9;
      const MIN_H     = 4;
      const MAX_H     = 56;

      this._waveformTimer = setInterval(() => {
        if (this._analyser) {
          const data = new Uint8Array(this._analyser.frequencyBinCount);
          this._analyser.getByteFrequencyData(data);

          // Map frequency bins to bar heights.
          const step = Math.floor(data.length / NUM_BARS);
          this.waveHeights = Array.from({ length: NUM_BARS }, (_, i) => {
            const bin = data[i * step] || 0;
            return MIN_H + Math.round((bin / 255) * (MAX_H - MIN_H));
          });
        } else {
          // Fallback: smooth random animation when analyser unavailable.
          this.waveHeights = this.waveHeights.map(h => {
            const delta = (Math.random() - 0.5) * 12;
            return Math.min(MAX_H, Math.max(MIN_H, h + delta));
          });
        }
      }, 80);  // ~12fps — smooth but not CPU-intensive
    },

    _stopWaveform() {
      clearInterval(this._waveformTimer);
      this._waveformTimer = null;
      this.waveHeights = Array(9).fill(8);  // reset to idle

      if (this._audioCtx) {
        this._audioCtx.close().catch(() => {});
        this._audioCtx = null;
        this._analyser = null;
      }
    },

    // ── Utilities ─────────────────────────────────────────────────────────

    /**
     * Format remaining seconds as M:SS for the countdown display.
     * @param {number} s  seconds
     * @returns {string}  e.g. "1:30" or "0:45"
     */
    formatTime(s) {
      const m = Math.floor(s / 60);
      const sec = s % 60;
      return `${m}:${String(sec).padStart(2, '0')}`;
    },

    _cleanup() {
      clearInterval(this._countdownTimer);
      clearInterval(this._waveformTimer);
      this._countdownTimer = null;
      this._waveformTimer  = null;

      if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
        try { this.mediaRecorder.stop(); } catch (_) {}
      }
      if (this.stream) {
        this.stream.getTracks().forEach(t => t.stop());
        this.stream = null;
      }
      if (this._audioCtx) {
        this._audioCtx.close().catch(() => {});
        this._audioCtx = null;
        this._analyser = null;
      }
    },

    // Called by Alpine when the component is destroyed (x-destroy).
    destroy() {
      this._cleanup();
    },
  };
}


/* ===========================================================================
   Utility: HTMX redirect handler
   =========================================================================== */

/**
 * HTMX does not natively follow HX-Redirect for full-page navigations
 * in all versions. This listener ensures a redirect header on any response
 * triggers a proper window.location change.
 */
document.addEventListener('htmx:beforeSwap', (event) => {
  const redirect = event.detail.xhr &&
                   event.detail.xhr.getResponseHeader('HX-Redirect');
  if (redirect) {
    event.preventDefault();      // stop HTMX from swapping DOM content
    window.location.href = redirect;
  }
});

/**
 * Suppress HTMX's default error alert (it shows a browser alert() dialog).
 * Errors are handled per-component via @htmx:response-error listeners.
 */
document.addEventListener('htmx:responseError', (event) => {
  event.preventDefault();
});
