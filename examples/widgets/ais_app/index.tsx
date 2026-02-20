// taze: setIframeSrcdocWithIntent,IframeIntent from //third_party/javascript/safevalues/dom
import {Marked} from 'marked';

// --- KATEX EXTENSIONS ---
interface KaTeX {
  renderToString(
    tex: string,
    options?: {displayMode?: boolean; throwOnError?: boolean},
  ): string;
}

interface WindowWithKatex extends Window {
  katex?: KaTeX;
}

interface KatexToken {
  text: string;
}

const katexBlock = {
  name: 'katex-block',
  level: 'block' as const,
  start(src: string) {
    return src.indexOf('$$');
  },
  tokenizer(src: string) {
    const match = /^\$\$([\s\S]+?)\$\$/.exec(src);
    if (match) {
      return {
        type: 'katex-block',
        raw: match[0],
        text: match[1].trim(),
      };
    }
    return undefined;
  },
  renderer(token: KatexToken) {
    const k = (window as WindowWithKatex).katex;
    return k
      ? k.renderToString(token.text, {displayMode: true, throwOnError: false})
      : token.text;
  },
};

const katexInline = {
  name: 'katex-inline',
  level: 'inline' as const,
  start(src: string) {
    return src.indexOf('$');
  },
  tokenizer(src: string) {
    const match = /^\$([^$\n]+?)\$/.exec(src);
    if (match) {
      return {
        type: 'katex-inline',
        raw: match[0],
        text: match[1].trim(),
      };
    }
    return undefined;
  },
  renderer(token: KatexToken) {
    const k = (window as WindowWithKatex).katex;
    return k
      ? k.renderToString(token.text, {displayMode: false, throwOnError: false})
      : token.text;
  },
};

const marked = new Marked().use({extensions: [katexBlock, katexInline]});

// Shim safevalues locally since they are missing in AI Studio runtime
// tslint:disable-next-line:no-unused-variable
const IframeIntent = {
  FORMATTED_HTML_CONTENT: 'FORMATTED_HTML_CONTENT',
};

// tslint:disable-next-line:no-unused-variable
function setIframeSrcdocWithIntent(
  iframe: HTMLIFrameElement,
  intent: unknown,
  content: string,
) {
  // tslint:disable-next-line:ban-ts-suppressions
  // @ts-ignore
  iframe.srcdoc = content;
}

// tslint:disable-next-line:no-unused-variable
function setElementInnerHtml(element: HTMLElement, html: unknown) {
  // tslint:disable-next-line:ban-ts-suppressions
  // @ts-ignore
  element.innerHTML = html as string;
}

// tslint:disable-next-line:no-unused-variable
function sanitizeHtmlAssertUnchanged(html: string) {
  return html;
}

// --- TYPES & CONSTANTS ---

interface Part {
  text?: string;
  inline_data?: {
    mime_type: string;
    data: string;
  };
  function_call?: {
    id: string;
    name: string;
    args: object;
  };
  function_response?: {
    id: string;
    response: unknown;
  };
}

interface FunctionResponseResult {
  function_response?: {
    parts?: Part[];
  };
}

interface FunctionResponseBody {
  result?: string | FunctionResponseResult;
  part?: Part;
  inline_data?: {
    mime_type: string;
    data: string;
  };
  data?: string;
  mime_type?: string;
}

interface FunctionResponse {
  id: string;
  response?: FunctionResponseBody | string;
  parts?: Part[];
  will_continue?: boolean;
}

interface MessagePayload {
  mimetype?: string;
  metadata: {
    is_final?: boolean;
  };
  part: Part;
  role: string;
  substream_name: string;
}

class ProcessorPart {
  constructor(
    public mimetype: string,
    // We avoid depending on TS GenAI SDK to keep the code simple.
    public part: Part,
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
  private onStatusChange: (status: ConnectionStatus) => void;
  private onMessage: (part: ProcessorPart) => void;

  constructor(
    onStatusChange: (status: ConnectionStatus) => void,
    onMessage: (part: ProcessorPart) => void,
  ) {
    this.onStatusChange = onStatusChange;
    this.onMessage = onMessage;
  }

  connect(url: string, config?: {[key: string]: unknown}) {
    if (this.socket) {
      console.log('Disconnecting existing socket before connect');
      this.disconnect();
    }

    try {
      this.onStatusChange(ConnectionStatus.CONNECTING);
      console.log(`Connecting to ${url}...`);
      this.socket = new WebSocket(url);

      this.socket.onopen = () => {
        console.log(
          'Socket onopen fired. readyState:',
          this.socket?.readyState,
        );
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

      this.socket.onclose = (event) => {
        console.log('Socket onclose fired:', event.code, event.reason);
        this.onStatusChange(ConnectionStatus.DISCONNECTED);
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket Error:', error);
        this.onStatusChange(ConnectionStatus.ERROR);
      };

      this.socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as MessagePayload;
          const mimetype = payload.mimetype || '';
          const isFinal =
            payload.metadata.is_final === undefined
              ? true
              : payload.metadata.is_final;

          this.onMessage(
            new ProcessorPart(
              mimetype,
              payload.part,
              isFinal,
              payload.role,
              payload.substream_name,
            ),
          );
        } catch (e) {
          console.error('Failed to parse or handle message:', e, event.data);
        }
      };
    } catch (e) {
      console.error('Connection threw error:', e);
      this.onStatusChange(ConnectionStatus.ERROR);
    }
  }

  disconnect() {
    if (this.socket) {
      console.log('Closing socket manually.');
      this.socket.close();
      this.socket = null;
    }
    this.onStatusChange(ConnectionStatus.DISCONNECTED);
  }

  sendText(text: string) {
    if (!this.socket) {
      console.error('SendText failed: socket is null');
      this.disconnect();
      return;
    }
    const message = {
      role: 'user',
      part: {
        text,
      },
    };
    this.socket.send(JSON.stringify(message));
  }
}

// --- DOM ELEMENTS & STATE ---

const el = {
  urlInput: document.getElementById('url-input') as HTMLInputElement,

  btnConnect: document.getElementById('btn-connect') as HTMLButtonElement,
  btnDisconnect: document.getElementById('btn-disconnect') as HTMLButtonElement,

  statusDot: document.getElementById('status-dot') as HTMLElement,
  statusText: document.getElementById('status-text') as HTMLElement,

  chatContainer: document.getElementById('chat-container') as HTMLElement,
  chatScrollArea: document.getElementById('chat-scroll-area') as HTMLElement,

  messageInput: document.getElementById('message-input') as HTMLInputElement,
  btnSend: document.getElementById('btn-send') as HTMLButtonElement,
  chatForm: document.getElementById('chat-form') as HTMLFormElement,
};

let currentBubbleId: string | null = null;
let currentRole: string | null = null;
let loadingBubbleId: string | null = null;
const funcCallBubbleIds: {[name: string]: string} = {};

// --- INITIALIZATION ---

const service = new StreamService(handleStatusChange, handleMessage);

// Force initial UI state
handleStatusChange(ConnectionStatus.DISCONNECTED);
console.log('WebSocket.OPEN constant is:', WebSocket.OPEN);

// Auto-connect on startup
connectToServer();

el.btnConnect.addEventListener('click', (e) => {
  e.preventDefault();
  connectToServer();
});

el.btnDisconnect.addEventListener('click', () => {
  service.disconnect();
});

el.chatForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = el.messageInput.value.trim();
  if (text) {
    // Optimistically show user message?
    // Or wait for server echo?
    // Usually servers echo input or we show it immediately.
    // For this rewrite, let's show it immediately as "committed".
    handleTextMessage(text, true, 'user', true);
    // Scroll so the user's message is at the top of the view
    scrollCurrentBubbleToTop();
    showLoadingIndicator();
    service.sendText(text);
    el.messageInput.value = '';
  }
});

// --- HANDLERS ---

function connectToServer() {
  const url = el.urlInput.value;
  service.connect(url);
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

  if (status === ConnectionStatus.CONNECTING) {
    el.chatContainer.textContent = '';
    currentBubbleId = null;
    currentRole = null;
    loadingBubbleId = null;
  }

  if (
    status === ConnectionStatus.DISCONNECTED ||
    status === ConnectionStatus.ERROR
  ) {
    hideLoadingIndicator();
  }

  el.statusText.textContent = status;

  // Update Controls
  const isConnected = status === ConnectionStatus.CONNECTED;
  const isConnecting = status === ConnectionStatus.CONNECTING;

  el.urlInput.disabled = isConnected || isConnecting;
  el.btnConnect.classList.toggle('hidden', isConnected);
  el.btnDisconnect.classList.toggle('hidden', !isConnected);

  el.messageInput.disabled = !isConnected;
  el.btnSend.disabled = !isConnected;

  if (!isConnected) {
    el.messageInput.placeholder = 'Connect to server to chat...';
  } else {
    el.messageInput.placeholder = 'Type your message...';
    el.messageInput.focus();
  }
}

function handleMessage(part: ProcessorPart) {
  if (part.substreamName === 'status') {
    return;
  }

  hideLoadingIndicator();

  if (part.part.text !== undefined) {
    handleTextMessage(part.part.text, part.isFinal, part.role);
  } else if (part.part.inline_data !== undefined) {
    handleImageMessage(
      part.part.inline_data.data,
      part.substreamName,
      part.mimetype,
      part.role,
    );
  } else if (part.part.function_call !== undefined) {
    handleFunctionCall(part.part.function_call);
  } else if (part.part.function_response !== undefined) {
    handleFunctionResponse(
      part.part.function_response as FunctionResponse,
      part.substreamName,
    );
  }
}

function showLoadingIndicator() {
  if (loadingBubbleId) return; // Already showing

  loadingBubbleId = `loading-${Math.random().toString(36).substring(7)}`;

  const wrapper = document.createElement('div');
  wrapper.className = 'animate-fade-in-up mb-4 flex flex-col items-start';
  wrapper.id = loadingBubbleId;

  const roleDiv = document.createElement('div');
  roleDiv.className =
    'text-xs font-bold uppercase tracking-wider mb-1 px-1 text-indigo-400';
  roleDiv.textContent = 'model';
  wrapper.appendChild(roleDiv);

  const messageDiv = document.createElement('div');
  messageDiv.className =
    'bg-gray-800/80 rounded-lg p-3 text-gray-100 text-sm leading-relaxed border border-gray-600/30 whitespace-pre-wrap max-w-[90%] md:max-w-[80%] flex items-center gap-1';

  const dots = createAnimatedDots();
  messageDiv.appendChild(dots);

  wrapper.appendChild(messageDiv);
  el.chatContainer.appendChild(wrapper);
}

function hideLoadingIndicator() {
  if (loadingBubbleId) {
    const element = document.getElementById(loadingBubbleId);
    if (element) {
      element.remove();
    }
    loadingBubbleId = null;
  }
}

function createAnimatedDots() {
  const container = document.createElement('div');
  container.className = 'flex items-center gap-1';
  for (let i = 0; i < 3; i++) {
    const dot = document.createElement('div');
    dot.className = `w-2 h-2 bg-gray-400 rounded-full animate-bounce`;
    dot.style.animationDelay = `${i * 150}ms`;
    container.appendChild(dot);
  }
  return container;
}

function renderPart(container: Element, part: Part) {
  if (part.inline_data) {
    if (part.inline_data.mime_type === 'text/html') {
      renderHtmlInIframe(container, part.inline_data.data);
    } else {
      renderImageInDiv(
        container,
        part.inline_data.data,
        part.inline_data.mime_type,
      );
    }
  } else if (part.text) {
    const textNode = document.createElement('div');
    textNode.className =
      'whitespace-pre-wrap font-mono text-xs text-cyan-100/90';
    textNode.textContent = part.text;
    container.appendChild(textNode);
  } else {
    const textNode = document.createElement('div');
    textNode.className =
      'whitespace-pre-wrap font-mono text-xs text-cyan-100/90';
    textNode.textContent = JSON.stringify(part, null, 2);
    container.appendChild(textNode);
  }
}

function handleFunctionCall(functionCall: {
  id: string;
  name: string;
  args: object;
}) {
  // Ensure we have a model bubble to append to
  if (currentRole !== 'model' || !currentBubbleId) {
    createBubble('model');
  }

  const bubble = document.getElementById(currentBubbleId!);
  if (!bubble) return; // Should not happen after createBubble

  // "Retire" existing markdown block so subsequent text goes to new block after the widget
  retireCurrentMarkdownBlock(bubble);

  const widgetId = `func-${Math.random().toString(36).substring(7)}`;

  // Store it to update later using the function call ID for correct matching
  funcCallBubbleIds[functionCall.id] = widgetId;
  // Note: We do NOT reset currentBubbleId or currentRole, maintaining the stream.

  // Create the widget container
  const messageDiv = document.createElement('div');
  messageDiv.id = widgetId;
  // Updated styles for embedded widget: full width, margin top/bottom
  messageDiv.className =
    'bg-slate-900/50 rounded-lg p-3 text-cyan-300 text-sm border border-cyan-800/50 w-full flex flex-col gap-2 shadow-inner my-2';

  // Title/Name
  const titleDiv = document.createElement('div');
  titleDiv.className = 'font-mono text-xs opacity-70 flex items-center gap-2';
  const iconSpan = document.createElement('span');
  iconSpan.className = 'text-cyan-500';
  iconSpan.textContent = '⚡';
  titleDiv.appendChild(iconSpan);
  titleDiv.appendChild(document.createTextNode(` ${functionCall.name}`));
  messageDiv.appendChild(titleDiv);

  // Content Area (initially just dots)
  const contentDiv = document.createElement('div');
  contentDiv.className = 'func-content min-h-[1.5em] flex items-center';
  contentDiv.appendChild(createAnimatedDots());

  messageDiv.appendChild(contentDiv);

  // Append widget to the main bubble
  bubble.appendChild(messageDiv);

  // Create new markdown block for subsequent model text
  appendMarkdownBlock(bubble);
}

function handleFunctionResponse(
  functionResponse: FunctionResponse,
  substreamName: string,
) {
  // We only show ui messages.
  if (substreamName !== 'ui') {
    return;
  }

  const bubbleId = funcCallBubbleIds[functionResponse.id];
  if (!bubbleId) return;

  const bubble = document.getElementById(bubbleId);
  if (!bubble) return;

  const contentDiv = bubble.querySelector('.func-content');
  if (!contentDiv) return;

  // Always remove dots if will_continue is false
  if (!functionResponse.will_continue) {
    const loading = contentDiv.querySelector('.animate-bounce');
    if (loading && loading.parentElement) {
      loading.parentElement.remove();
    }
  }

  // Check if we are still showing dots (loading state)
  const loading = contentDiv.querySelector('.animate-bounce');
  if (loading && loading.parentElement) {
    loading.parentElement.remove(); // Remove only the dots container
  }

  // Handle different response formats
  contentDiv.className = 'func-content w-full'; // Remove flex/center to allow full width content

  let processed = false;

  if (functionResponse.response) {
    const response = functionResponse.response;

    if (
      response &&
      typeof response === 'object' &&
      'result' in response &&
      response.result
    ) {
      const result = response.result;
      if (typeof result === 'string') {
        if (result.length > 0) {
          const textNode = document.createElement('div');
          textNode.className =
            'whitespace-pre-wrap font-mono text-xs text-cyan-100/90';
          textNode.textContent = result;
          contentDiv.appendChild(textNode);
        }
        processed = true;
      } else if (result && typeof result === 'object') {
        const resObj = result as FunctionResponseResult;
        if (
          resObj.function_response &&
          Array.isArray(resObj.function_response.parts)
        ) {
          for (const part of resObj.function_response.parts) {
            renderPart(contentDiv, part);
          }
          processed = true;
        }
      }
    }

    if (!processed) {
      if (typeof response === 'string') {
        const textNode = document.createElement('div');
        textNode.className =
          'whitespace-pre-wrap font-mono text-xs text-cyan-100/90';
        textNode.textContent = response;
        contentDiv.appendChild(textNode);
      } else {
        const anyResp = response as FunctionResponseBody;
        if (anyResp.part) {
          renderPart(contentDiv, anyResp.part);
        } else if (anyResp.inline_data || anyResp.data) {
          const data = anyResp.inline_data?.data || anyResp.data;
          const mime =
            anyResp.inline_data?.mime_type || anyResp.mime_type || 'image/png';
          if (data) {
            renderPart(contentDiv, {
              inline_data: {data, mime_type: mime},
            });
          } else {
            const textNode = document.createElement('div');
            textNode.className =
              'whitespace-pre-wrap font-mono text-xs text-cyan-100/90';
            textNode.textContent = JSON.stringify(response, null, 2);
            contentDiv.appendChild(textNode);
          }
        } else {
          const textNode = document.createElement('div');
          textNode.className =
            'whitespace-pre-wrap font-mono text-xs text-cyan-100/90';
          textNode.textContent = JSON.stringify(response, null, 2);
          contentDiv.appendChild(textNode);
        }
      }
    }
  } else if (functionResponse.parts) {
    for (const part of functionResponse.parts) {
      renderPart(contentDiv, part);
    }
  } else {
    // Fallback entire object
    const textNode = document.createElement('div');
    textNode.className =
      'whitespace-pre-wrap font-mono text-xs text-cyan-100/90';
    textNode.textContent = JSON.stringify(functionResponse, null, 2);
    contentDiv.appendChild(textNode);
  }

  // Check for will_continue to re-add dots
  if (functionResponse.will_continue) {
    const dots = createAnimatedDots();
    contentDiv.appendChild(dots);
  }
}

const IFRAME_MAX_HEIGHT = 1500;

function resizeIframeToContent(iframe: HTMLIFrameElement) {
  try {
    const doc = iframe.contentDocument || iframe.contentWindow?.document;
    if (doc && doc.body) {
      const contentHeight = doc.body.scrollHeight;
      const newHeight = Math.min(contentHeight + 16, IFRAME_MAX_HEIGHT);
      iframe.style.height = `${newHeight}px`;
    }
  } catch (e) {
    console.error('Failed to resize iframe', e);
  }
}

function renderHtmlInIframe(container: Element, data: string) {
  // Decode base64 to string
  let decoded: string;
  try {
    decoded = atob(data);
  } catch (e) {
    console.error('Failed to decode base64 html', e);
    decoded = 'Error decoding HTML content.';
  }

  // Check if the last child is already an iframe - if so, append to it
  const lastChild = container.lastElementChild;
  if (lastChild && lastChild.tagName === 'IFRAME') {
    const existingIframe = lastChild as HTMLIFrameElement;
    const existingContent = existingIframe.srcdoc || '';
    setIframeSrcdocWithIntent(
      existingIframe,
      IframeIntent.FORMATTED_HTML_CONTENT,
      existingContent + decoded,
    );
    // Resize after content update - use setTimeout to let content render
    setTimeout(() => resizeIframeToContent(existingIframe), 50);
    return;
  }

  // Create new iframe
  const iframe = document.createElement('iframe');
  iframe.className = 'rounded-md border border-cyan-900/50 mt-2 bg-white';
  iframe.style.width = '100%';
  iframe.style.height = '320px';

  // Resize on load
  iframe.onload = () => resizeIframeToContent(iframe);

  setIframeSrcdocWithIntent(
    iframe,
    IframeIntent.FORMATTED_HTML_CONTENT,
    decoded,
  );

  container.appendChild(iframe);
}

function renderImageInDiv(container: Element, data: string, mimetype: string) {
  const img = document.createElement('img');
  const src = data.startsWith('data:')
    ? data
    : `data:${mimetype};base64,${data}`;
  img.src = src;
  img.className = 'w-full h-auto rounded-md border border-cyan-900/50 mt-2';
  container.appendChild(img);
}

function handleTextMessage(
  content: string,
  isFinal: boolean,
  role = 'user',
  forceNewBubble = false,
) {
  // Treat 'tool' role as 'model' to group them in the same bubble
  if (role === 'tool') {
    role = 'model';
  }

  const safeRole = role.toLowerCase();

  // If role changes or no bubble exists, create a new one
  if (forceNewBubble || currentRole !== safeRole || !currentBubbleId) {
    createBubble(safeRole);
  }

  const bubble = document.getElementById(currentBubbleId!);
  if (!bubble) return;

  const markdownBlock = bubble.querySelector(
    '.markdown-content:last-of-type',
  ) as HTMLElement;
  if (!markdownBlock) {
    // Should typically exist from createBubble, but safety check
    appendMarkdownBlock(bubble);
    return handleTextMessage(content, isFinal, role, false);
  }

  // We store the raw markdown in a data attribute or separate hidden property
  // to support streaming accumulation.
  // "committed" text is what has been finalized.
  // "tentative" text is the current chunk if not final.

  // Note: we're simplifying the span approach to a single block for markdown.
  // We need to store state on the element to know what is committed.
  let committed = markdownBlock.dataset['committed'] || '';
  let tentative = '';

  if (isFinal) {
    committed += content;
    markdownBlock.dataset['committed'] = committed;
  } else {
    tentative = content;
  }

  const fullMarkdown = committed + tentative;

  // Render markdown
  // marked.parse returns a string (Promise if async, but default is sync)
  const html = marked.parse(fullMarkdown, {async: false});
  setElementInnerHtml(markdownBlock, sanitizeHtmlAssertUnchanged(html));
}

function handleImageMessage(
  data: string,
  substream: string,
  mimetype: string,
  role: string,
) {
  currentBubbleId = null; // force next text to start new bubble

  const wrapper = document.createElement('div');
  wrapper.className = 'animate-fade-in-up mb-4 flex flex-col';

  if (role === 'user') wrapper.classList.add('items-end');
  else wrapper.classList.add('items-start');

  const roleDiv = document.createElement('div');
  roleDiv.className = `text-xs font-bold uppercase tracking-wider mb-1 px-1 ${
    role === 'model' ? 'text-indigo-400' : 'text-amber-500'
  }`;
  roleDiv.textContent = role;
  wrapper.appendChild(roleDiv);

  const imgContainer = document.createElement('div');
  imgContainer.className =
    'max-w-[80%] rounded-xl overflow-hidden border border-gray-700 shadow-xl';

  const src = data.startsWith('data:')
    ? data
    : `data:${mimetype};base64,${data}`;
  const img = document.createElement('img');
  img.src = src;
  img.className = 'w-full h-auto';

  imgContainer.appendChild(img);
  wrapper.appendChild(imgContainer);

  el.chatContainer.appendChild(wrapper);
}

function retireCurrentMarkdownBlock(bubble: HTMLElement) {
  const block = bubble.querySelector('.markdown-content:last-of-type');
  if (block) {
    block.classList.remove('markdown-content');
    block.classList.add('markdown-content-retired');
  }
}

function appendMarkdownBlock(container: HTMLElement) {
  const block = document.createElement('div');
  // prose-invert for dark mode markdown styling if using tailwind typography,
  // or just custom styles. We'll stick to basic styling inherited or simple overrides.
  // We add 'markdown-content' class to identify it as the active append target.
  block.className =
    'markdown-content prose prose-invert max-w-none text-sm leading-relaxed';
  container.appendChild(block);
}

function createBubble(role: string) {
  currentRole = role;
  currentBubbleId = `msg-${Math.random().toString(36).substring(7)}`;

  const wrapper = document.createElement('div');
  wrapper.className = 'animate-fade-in-up mb-4 flex flex-col';
  if (role === 'user') {
    wrapper.classList.add('items-end');
  } else {
    wrapper.classList.add('items-start');
  }

  // Role Header
  const roleDiv = document.createElement('div');
  roleDiv.className = `text-xs font-bold uppercase tracking-wider mb-1 px-1 ${
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

  const bgClass = role === 'user' ? 'bg-gray-700/80' : 'bg-gray-800/80';
  const textClass = role === 'user' ? 'text-amber-100' : 'text-gray-100';

  messageDiv.className = `${bgClass} rounded-lg p-3 ${textClass} text-sm leading-relaxed border border-gray-600/30 whitespace-pre-wrap max-w-[90%] md:max-w-[80%]`;

  wrapper.appendChild(messageDiv);
  el.chatContainer.appendChild(wrapper);

  // Add initial markdown block
  appendMarkdownBlock(messageDiv);
}

function scrollCurrentBubbleToTop() {
  if (currentBubbleId) {
    const bubble = document.getElementById(currentBubbleId);
    if (bubble && bubble.parentElement) {
      // The parent wrapper contains the role label, scroll to that
      const wrapper = bubble.parentElement;
      wrapper.scrollIntoView({block: 'start', behavior: 'instant'});
    }
  }
}
