// --- TYPES & CONSTANTS ---

class ProcessorPart {
  constructor(
    public mimetype: string,
    // We avoid depending on TS GenAI SDK to keep the code simple.
    // tslint:disable-next-line:no-any
    public part: any,
    public isFinal: boolean,
    public role: string,
    public substreamName: string,
  ) {}
}

enum ConnectionStatus {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  ERROR = 'error',
}

// --- STREAM SERVICE ---

class StreamService {
  private socket: WebSocket | null = null;
  private audioContext: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private mediaStream: MediaStream | null = null;
  private input: MediaStreamAudioSourceNode | null = null;

  private onStatusChange: (status: ConnectionStatus) => void;
  private onMessage: (part: ProcessorPart) => void;
  private targetSampleRate = 24000;

  constructor(
    onStatusChange: (status: ConnectionStatus) => void,
    onMessage: (part: ProcessorPart) => void,
  ) {
    this.onStatusChange = onStatusChange;
    this.onMessage = onMessage;
  }

  connect(url: string, config?: {[key: string]: unknown}) {
    if (this.socket) this.disconnect();

    try {
      this.onStatusChange(ConnectionStatus.CONNECTING);
      this.socket = new WebSocket(url);

      this.socket.onopen = () => {
        this.onStatusChange(ConnectionStatus.CONNECTED);
        if (config) {
          this.socket?.send(
            JSON.stringify({
              mimetype: 'application/x-config',
              metadata: config,
            }),
          );
        }
      };

      this.socket.onclose = () => {
        this.onStatusChange(ConnectionStatus.DISCONNECTED);
        this.stopAudio();
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket Error:', error);
        this.onStatusChange(ConnectionStatus.ERROR);
      };

      this.socket.onmessage = (event) => {
        try {
          // tslint:disable-next-line:no-any
          const payload = JSON.parse(event.data) as {[key: string]: any};
          const mimetype = payload['mimetype'] || '';
          const isFinal =
            payload['metadata'].is_final === undefined
              ? true
              : payload['metadata'].is_final;

          this.onMessage(
            new ProcessorPart(
              mimetype,
              payload['part'],
              isFinal,
              payload['role'],
              payload['substream_name'],
            ),
          );
        } catch (e) {
          console.error('Failed to parse message:', event.data);
        }
      };
    } catch (e) {
      this.onStatusChange(ConnectionStatus.ERROR);
    }
  }

  disconnect() {
    this.stopAudio();
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    this.onStatusChange(ConnectionStatus.DISCONNECTED);
  }

  async startAudio() {
    if (!this.socket || this.socket.readyState !== 1) {
      throw new Error('Socket not connected');
    }

    this.mediaStream = await navigator.mediaDevices.getUserMedia({audio: true});

    const AudioContextClass = window.AudioContext;
    this.audioContext = new AudioContextClass({
      sampleRate: this.targetSampleRate,
    });

    this.input = this.audioContext.createMediaStreamSource(this.mediaStream);
    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

    this.processor.onaudioprocess = (e) => {
      if (!this.socket || this.socket.readyState !== 1) return;

      const inputData = e.inputBuffer.getChannelData(0);
      const pcmData = this.floatTo16BitPCM(inputData);
      const base64Audio = this.arrayBufferToBase64(pcmData.buffer);

      const message = {
        part: {
          inline_data: {
            data: base64Audio,
            mime_type: 'audio/l16;rate=24000',
          },
        },
        role: 'user',
        substream_name: 'realtime',
        mimetype: 'audio/l16;rate=24000',
      };
      this.socket.send(JSON.stringify(message));
    };

    this.input.connect(this.processor);
    this.processor.connect(this.audioContext.destination);
  }

  stopAudio() {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    if (this.input) {
      this.input.disconnect();
      this.input = null;
    }
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }

  private floatTo16BitPCM(input: Float32Array): Int16Array {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]));
      output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return output;
  }

  private arrayBufferToBase64(buffer: ArrayBuffer): string {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
  }
}

// --- DOM ELEMENTS & STATE ---

const el = {
  urlInput: document.getElementById('url-input') as HTMLInputElement,
  imageSystemInstructionInput: document.getElementById(
    'image-system-instruction-input',
  ) as HTMLInputElement,
  imagePeriodInput: document.getElementById(
    'image-period-input',
  ) as HTMLInputElement,
  settingsPanel: document.getElementById('settings-panel') as HTMLElement,
  btnToggleSettings: document.getElementById(
    'btn-toggle-settings',
  ) as HTMLButtonElement,
  iconChevron: document.getElementById('icon-chevron') as HTMLElement,
  btnConnect: document.getElementById('btn-connect') as HTMLButtonElement,
  btnDisconnect: document.getElementById('btn-disconnect') as HTMLButtonElement,
  btnReset: document.getElementById('btn-reset') as HTMLButtonElement,
  btnMic: document.getElementById('btn-mic') as HTMLButtonElement,
  micIcon: document.getElementById('mic-icon') as HTMLElement,
  micIconActive: document.getElementById('mic-icon-active') as HTMLElement,
  micText: document.getElementById('mic-text') as HTMLElement,

  statusDot: document.getElementById('status-dot') as HTMLElement,
  statusText: document.getElementById('status-text') as HTMLElement,

  textContainer: document.getElementById('text-container') as HTMLElement,
  textScrollArea: document.getElementById('text-scroll-area') as HTMLElement,
  textCount: document.getElementById('text-count') as HTMLElement,
  textEmptyState: document.getElementById('text-empty-state') as HTMLElement,

  imageContainer: document.getElementById('image-container') as HTMLElement,
  imageCount: document.getElementById('image-count') as HTMLElement,
  imageEmptyState: document.getElementById('image-empty-state') as HTMLElement,
  conceptArtToggle: document.getElementById(
    'concept-art-toggle',
  ) as HTMLInputElement,
  btnCopyImages: document.getElementById(
    'btn-copy-images',
  ) as HTMLButtonElement,
  copyTooltip: document.getElementById('copy-tooltip') as HTMLElement,
};

let isRecording = false;
let messageCount = 0;
let imageCount = 0;
let currentBubbleId: string | null = null;
let currentRole: string | null = null;

// --- INITIALIZATION ---

const service = new StreamService(handleStatusChange, handleMessage);

el.btnConnect.addEventListener('click', (e) => {
  e.preventDefault();
  connectToServer();
  el.settingsPanel.classList.add('hidden');
  el.iconChevron.classList.remove('rotate-180');
});

el.btnDisconnect.addEventListener('click', () => {
  service.disconnect();
  setRecordingState(false);
});

el.imageSystemInstructionInput.addEventListener('input', () => {
  service.disconnect();
  setRecordingState(false);
});

el.imagePeriodInput.addEventListener('input', () => {
  service.disconnect();
  setRecordingState(false);
});

el.btnMic.addEventListener('click', () => {
  toggleRecording();
});

el.btnToggleSettings.addEventListener('click', () => {
  el.settingsPanel.classList.toggle('hidden');
  el.iconChevron.classList.toggle('rotate-180');
});

el.conceptArtToggle.addEventListener('change', (e) => {
  const isChecked = (e.target as HTMLInputElement).checked;
  const images = document.querySelectorAll('.substream_create_concept_art');
  images.forEach((img) => {
    if (isChecked) {
      (img as HTMLElement).style.display = '';
    } else {
      (img as HTMLElement).style.display = 'none';
    }
  });
});

el.btnCopyImages.addEventListener('click', () => {
  copyImagesToClipboard();
});

el.btnReset.addEventListener('click', () => {
  reset();
});

// --- HANDLERS ---

function connectToServer() {
  const url = el.urlInput.value;
  const imageInstructions = el.imageSystemInstructionInput.value.trim();
  const imagePeriod = Number(el.imagePeriodInput.value.trim());
  const config: {[key: string]: unknown} = {};
  if (imageInstructions) config['image_system_instruction'] = imageInstructions;
  if (!isNaN(imagePeriod)) config['image_period_sec'] = imagePeriod;
  service.connect(url, config);
}

function reset() {
  // Clear text and image containers
  el.textContainer.textContent = '';
  el.imageContainer.textContent = '';

  // Show empty states
  el.textEmptyState.classList.remove('hidden');
  el.imageEmptyState.classList.remove('hidden');

  // Reset counts
  messageCount = 0;
  imageCount = 0;
  el.textCount.textContent = '0 turns';
  el.imageCount.textContent = '0 images';

  // Reset bubble state
  currentBubbleId = null;
  currentRole = null;

  // Reconnect
  service.disconnect();
  connectToServer();
}

function handleStatusChange(status: ConnectionStatus) {
  // Update Dot
  el.statusDot.className = `w-2.5 h-2.5 rounded-full transition-all duration-300 ${
    status === ConnectionStatus.CONNECTED
      ? 'bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.5)]'
      : status === ConnectionStatus.CONNECTING
        ? 'bg-yellow-500 animate-pulse'
        : status === ConnectionStatus.ERROR
          ? 'bg-red-500'
          : 'bg-gray-500'
  }`;

  el.statusText.textContent = status;

  // Update Controls
  const isConnected = status === ConnectionStatus.CONNECTED;
  const isConnecting = status === ConnectionStatus.CONNECTING;

  el.urlInput.disabled = isConnected || isConnecting;
  el.btnConnect.disabled = isConnecting;
  el.btnConnect.textContent = isConnecting ? 'Connecting...' : 'Connect';

  if (isConnected) {
    el.btnConnect.classList.add('hidden');
    el.btnDisconnect.classList.remove('hidden');
    el.btnReset.classList.remove('hidden');

    // Enable Mic
    el.btnMic.disabled = false;
    el.btnMic.classList.remove('opacity-50', 'cursor-not-allowed');
    el.btnMic.classList.add('hover:bg-gray-600');
    if (!isRecording) {
      toggleRecording();
    }
  } else {
    el.btnConnect.classList.remove('hidden');
    el.btnDisconnect.classList.add('hidden');
    el.btnReset.classList.add('hidden');

    // Disable Mic
    setRecordingState(false);
    el.btnMic.disabled = true;
    el.btnMic.className =
      'flex items-center justify-center px-6 py-2.5 rounded-lg border transition-all duration-200 text-sm font-bold bg-gray-700 border-gray-600 text-gray-500 opacity-50 cursor-not-allowed';
  }
}

async function toggleRecording() {
  if (isRecording) {
    service.stopAudio();
    setRecordingState(false);
  } else {
    try {
      await service.startAudio();
      setRecordingState(true);
    } catch (err) {
      console.error(err);
      alert('Could not access microphone.');
      setRecordingState(false);
    }
  }
}

function setRecordingState(recording: boolean) {
  isRecording = recording;

  if (recording) {
    el.micIcon.classList.add('hidden');
    el.micIconActive.classList.remove('hidden');
    el.micText.textContent = 'ON';

    // Active Styles
    el.btnMic.classList.remove(
      'bg-gray-700',
      'text-gray-300',
      'hover:bg-gray-600',
    );
    el.btnMic.classList.add(
      'bg-red-500/20',
      'border-red-500',
      'text-red-500',
      'hover:bg-red-500/30',
      'shadow-[0_0_10px_rgba(239,68,68,0.3)]',
    );
  } else {
    el.micIcon.classList.remove('hidden');
    el.micIconActive.classList.add('hidden');
    el.micText.textContent = 'MIC';

    // Inactive Styles (only if connected, which checks occur elsewhere, but safe to reset base styles here if connected)
    if (!el.btnMic.disabled) {
      el.btnMic.classList.remove(
        'bg-red-500/20',
        'border-red-500',
        'text-red-500',
        'hover:bg-red-500/30',
        'shadow-[0_0_10px_rgba(239,68,68,0.3)]',
      );
      el.btnMic.classList.add(
        'bg-gray-700',
        'border-gray-600',
        'text-gray-300',
        'hover:bg-gray-600',
      );
    }
  }
}

function handleMessage(part: ProcessorPart) {
  if (part.substreamName === 'status') {
    // Do not show status messages as they clutter the output.
    return;
  }

  if (part.part.text !== undefined) {
    handleTextMessage(part.part.text, part.isFinal, part.role);
  } else if (part.part.inline_data !== undefined) {
    handleImageMessage(
      part.part.inline_data.data,
      part.substreamName,
      part.mimetype,
    );
  } else if (part.part.function_call !== undefined) {
    handleFunctionCall(part.part.function_call);
  }
}

function handleFunctionCall(functionCall: {name: string; args: object}) {
  if (functionCall.name === 'create_concept_art') {
    const args = functionCall.args as {description: string; name: string};
    handleTextMessage(
      args.description,
      true,
      `🖌️ concept art(${args.name})`,
      true,
    );
  } else if (functionCall.name === 'create_image_from_description') {
    const args = functionCall.args as {description: string; concept_arts: unknown[]};
    const concept_arts = JSON.stringify(args.concept_arts);
    handleTextMessage(
      `${args.description}\n${concept_arts}`,
      true,
      '🖌️ illustrate',
      true,
    );
  } else {
    const role = `🔧 ${functionCall.name}`;
    const content = JSON.stringify(functionCall.args, null, 2);
    handleTextMessage(content, true, role, true);
  }
}

function handleTextMessage(
  content: string,
  isFinal: boolean,
  role = 'user',
  forceNewBubble = false,
) {
  if (role === 'tool') {
    // To avoid disrupting the transcrption in the UI, we do not show text
    // output from Nano Banana.
    return;
  }

  el.textEmptyState.classList.add('hidden');
  const safeRole = role.toLowerCase();

  // If role changes or no bubble exists, create a new one
  if (forceNewBubble || currentRole !== safeRole || !currentBubbleId) {
    createBubble(safeRole);
  }

  const bubble = document.getElementById(currentBubbleId!);
  if (!bubble) return;

  const committedSpan = bubble.querySelector('.committed-text');
  const tentativeSpan = bubble.querySelector('.tentative-text');

  if (committedSpan && tentativeSpan) {
    if (isFinal) {
      // Append final text to the committed section
      // Add a space if there is already committed text
      const prefix = committedSpan.textContent ? ' ' : '';
      committedSpan.textContent += prefix + content;
      tentativeSpan.textContent = ''; // Clear tentative
    } else {
      // Update tentative text (streaming/partial)
      tentativeSpan.textContent = content;
    }
  }

  // Update timestamp
  const wrapper = bubble.parentElement;
  const ts = wrapper?.querySelector('.timestamp');
  if (ts) ts.textContent = new Date().toLocaleTimeString();

  scrollToBottom(el.textScrollArea);
}

function createBubble(role: string) {
  currentRole = role;
  currentBubbleId = `msg-${Math.random().toString(36).substring(7)}`;

  messageCount++;
  el.textCount.textContent = `${messageCount} turns`;

  const wrapper = document.createElement('div');
  wrapper.className = 'animate-fade-in-up mb-6';

  // Role Header
  const roleDiv = document.createElement('div');
  roleDiv.className = `text-xs font-bold uppercase tracking-wider mb-1 ml-1 ${
    role === 'model'
      ? 'text-indigo-400'
      : role === 'tool'
        ? 'text-green-400'
        : 'text-amber-500'
  }`;
  roleDiv.textContent = role;
  wrapper.appendChild(roleDiv);

  // Bubble Container
  const messageDiv = document.createElement('div');
  messageDiv.id = currentBubbleId;
  const textColorClass = role === 'user' ? 'text-amber-200' : 'text-gray-200';
  messageDiv.className = `bg-gray-700/50 rounded-lg p-3 ${textColorClass} text-sm leading-relaxed border border-gray-600/30 whitespace-pre-wrap`;

  // Spans for committed vs tentative text
  const committedSpan = document.createElement('span');
  committedSpan.className = 'committed-text';

  const tentativeSpan = document.createElement('span');
  tentativeSpan.className = 'tentative-text opacity-80'; // Tentative text slightly transparent

  messageDiv.appendChild(committedSpan);
  messageDiv.appendChild(tentativeSpan);
  wrapper.appendChild(messageDiv);

  // Timestamp
  const timestampDiv = document.createElement('div');
  timestampDiv.className =
    'timestamp text-[10px] text-gray-500 mt-1 px-1 text-right font-mono';
  timestampDiv.textContent = new Date().toLocaleTimeString();
  wrapper.appendChild(timestampDiv);

  el.textContainer.appendChild(wrapper);
}

function handleImageMessage(
  data: string,
  substream_name: string,
  mimetype: string,
) {
  const content = data.startsWith('data:')
    ? data
    : `data:${mimetype};base64,${data}`;
  el.imageEmptyState.classList.add('hidden');
  imageCount++;
  el.imageCount.textContent = `${imageCount} images`;

  // Remove "Newest" badge from previous newest image and demote its styles
  const oldNewestBadge = el.imageContainer.querySelector('.badge-newest');
  if (oldNewestBadge) {
    const previousNewestImage = oldNewestBadge.parentElement;
    oldNewestBadge.remove();
    if (previousNewestImage) {
      previousNewestImage.classList.remove(
        'ring-2',
        'ring-primary-500',
        'scale-100',
      );
      previousNewestImage.classList.add(
        'opacity-80',
        'scale-95',
        'hover:opacity-100',
        'hover:scale-[0.97]',
      );
    }
  }

  const div = document.createElement('div');
  div.className =
    'relative group rounded-xl overflow-hidden shadow-xl border border-gray-700 transition-transform duration-500 ring-2 ring-primary-500 scale-100 animate-fade-in-up';
  div.className += ' substream_' + substream_name;

  if (substream_name === 'create_concept_art' && !el.conceptArtToggle.checked) {
    div.style.display = 'none';
  }

  const badgeDiv = document.createElement('div');
  badgeDiv.className =
    'badge-newest absolute top-2 right-2 bg-primary-600 text-white text-[10px] font-bold px-2 py-0.5 rounded-full shadow-lg z-10 uppercase tracking-wide';
  badgeDiv.textContent = 'Newest';
  div.appendChild(badgeDiv);

  if (substream_name === 'create_concept_art') {
    const conceptArtBadge = document.createElement('div');
    conceptArtBadge.className =
      'concept-art-badge absolute top-2 left-2 bg-purple-600 text-white text-[10px] font-bold px-2 py-0.5 rounded-full shadow-lg z-10 uppercase tracking-wide';
    conceptArtBadge.textContent = 'Concept Art';
    div.appendChild(conceptArtBadge);
  }

  const img = document.createElement('img');
  img.src = content;
  img.className = 'max-h-[75vh] w-auto h-auto mx-auto bg-gray-900 max-w-full';
  img.loading = 'lazy';
  div.appendChild(img);

  const timeDiv = document.createElement('div');
  timeDiv.className =
    'timestamp-div absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent p-4 opacity-0 group-hover:opacity-100 transition-opacity';
  const timeSpan = document.createElement('span');
  timeSpan.className = 'text-xs text-gray-300 font-mono';
  timeSpan.textContent = new Date().toLocaleTimeString();
  timeDiv.appendChild(timeSpan);
  div.appendChild(timeDiv);

  el.imageContainer.append(div);
  div.scrollIntoView();
  img.onload = () => {
    div.scrollIntoView();
  };
}

function showCopyTooltip() {
  el.copyTooltip.classList.remove('hidden');
  setTimeout(() => {
    el.copyTooltip.classList.add('hidden');
  }, 3000);
}

async function copyImagesToClipboard() {
  const images = Array.from(el.imageContainer.children);
  const conceptArtImages: HTMLElement[] = [];
  const illustrationImages: HTMLElement[] = [];

  images.forEach((node) => {
    const image = node.cloneNode(true) as HTMLElement;
    image.querySelectorAll('.timestamp-div').forEach((el) => el.remove());
    image.querySelectorAll('.badge-newest').forEach((el) => el.remove());
    image.querySelectorAll('.concept-art-badge').forEach((el) => el.remove());

    if (image.classList.contains('substream_create_concept_art')) {
      image.style.display = '';
      conceptArtImages.push(image);
    } else {
      illustrationImages.push(image);
    }
  });

  let html = '';
  if (conceptArtImages.length > 0) {
    html += '<h1>Concept Art</h1>';
    html += conceptArtImages.map((img) => img.outerHTML).join('');
  }
  if (illustrationImages.length > 0) {
    html += '<h1>Illustrations</h1>';
    html += illustrationImages.map((img) => img.outerHTML).join('');
  }

  const clipboardItem = new ClipboardItem({
    'text/html': new Blob([html], {type: 'text/html'}),
  });
  await navigator.clipboard.write([clipboardItem]);
  showCopyTooltip();
}

function scrollToBottom(element: HTMLElement) {
  element.scrollTop = element.scrollHeight;
}
