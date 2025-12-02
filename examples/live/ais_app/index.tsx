/**
 * @fileoverview Live Commentator Agent client.
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import * as aiae from '@google/actionengine';
import '@material/web/icon/icon.js';
import '@material/web/iconbutton/icon-button.js';
import {css, html, LitElement, PropertyValueMap} from 'lit';
import {customElement, query, state} from 'lit/decorators.js';

// Address of WebSocket server running the agent, e.g. "ws://localhost:8765".
const AGENT_WEBSOCKET_URL = 'ws://localhost:8765';

@customElement('live-commentator')
export class LiveCommentator extends LitElement {
  @state() accessor mic = false;
  @state() accessor video = '';
  @state() videoSources = {};
  @state() accessor screen = false;
  @state() accessor transcript = '';
  @state() accessor showTranscript = true;
  @state() accessor chattiness = 0.5;

  @query('.audio-out') accessor audioOut!: HTMLAudioElement;
  @query('.video-out') accessor videoOut!: HTMLVideoElement;
  @query('.screen-out') accessor screenOut!: HTMLVideoElement;
  @query('.transcript-box') accessor transcriptBox!: HTMLDivElement;

  private micStream?: MediaStream;
  private videoStream?: MediaStream;
  private screenStream?: MediaStream;
  private agentWebSocket?: WebSocket;
  private audioOutStream?: aiae.stream.Stream<aiae.Chunk>;

  override firstUpdated(props: PropertyValueMap<unknown>): void {
    super.firstUpdated(props);
    this.initializeVideoSources();
    this.initializeWebSocket();
    this.initializeAudioOut();
    this.transcriptOn();

    // First interaction trigger playing audio.
    this.shadowRoot?.children[0].addEventListener(
      'click',
      () => {
        void this.audioOut.play();
      },
      {once: true},
    );
    this.getRootNode().addEventListener('keydown', (e: Event) =>
      this.keydownHandler(e),
    );
  }

  private keydownHandler(e: Event) {
    const code = (e as KeyboardEvent).code;
    if (code === 'KeyM') {
      if (this.mic) this.micOff();
      else void this.micOn();
    }
    if (code === 'KeyR') {
      this.reset();
    }
    if (code === 'Digit1') {
      this.videoOn(Object.keys(this.videoSources)[0]);
    }
    if (code === 'Digit2') {
      this.videoOn(Object.keys(this.videoSources)[1]);
    }
    if (code === 'Digit3') {
      this.videoOn(Object.keys(this.videoSources)[2]);
    }
  }

  private initializeWebSocket() {
    this.agentWebSocket = new WebSocket(AGENT_WEBSOCKET_URL);
    this.agentWebSocket.onopen = () => {
      console.log('Connection to ' + AGENT_WEBSOCKET_URL + ' open.');
      this.sendChattiness();
    };
    this.agentWebSocket.onmessage = (event) => {
      const jsonData = JSON.parse(event.data) as {[key: string]: any};
      const mime_type = jsonData['mimetype'];
      if (mime_type?.startsWith('audio/')) {
        this.audioOutStream.write({
          data: aiae.base64.decode(jsonData['part']['inline_data']['data']),
          metadata: {mimetype: {type: 'audio', subtype: 'pcm'}},
        });
      } else if (mime_type?.startsWith('text/')) {
        this.transcript += jsonData['part']['text'];
      } else if (
        mime_type === 'application/x-state' &&
        (jsonData['metadata']['generation_complete'] ||
          jsonData['metadata']['interrupted'])
      ) {
        // Reset the transcript after a short delay.
        requestAnimationFrame(() => {
          setTimeout(() => {
            this.transcript = '';
          }, 500);
        });
      }
    };
    // If the connection is lost, keep trying to restart it.
    this.agentWebSocket.onclose = () => {
      this.initializeWebSocket();
      this.initializeAudioOut();
      this.transcript = '';
    };
  }

  private initializeAudioOut() {
    this.audioOutStream = aiae.stream.createStream<aiae.Chunk>();
    const media = aiae.content.audioChunksToMediaStream(
      this.audioOutIterable(),
    );
    this.audioOut.srcObject = media;
  }

  private async initializeVideoSources() {
    // For enumerateDevices to return video sources we need to have permissions to
    // use video inputs in the first place. This will show the permission dialog
    // if needed.
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    stream.getTracks().forEach((track) => track.stop());

    const devices = await navigator.mediaDevices.enumerateDevices();
    let videoSources = {};
    devices.forEach(function (device) {
      if (device.kind == 'videoinput') {
        videoSources[device.deviceId] = device.label;
      }
    });
    this.videoSources = videoSources;
  }

  private send(data: Object) {
    // Tries to send data to WebSocket, but ignores errors:
    // If the connection breaks we will be trying to reestablish it and resume
    // streaming the currently selected sources.
    try {
      this.agentWebSocket?.send(JSON.stringify(data));
    } catch (e) {
      console.log((e as Error).message);
    }
  }

  private async *audioOutIterable() {
    for await (const audioChunk of this.audioOutStream) {
      yield audioChunk;
    }
  }

  private async micOn() {
    this.mic = true;
    this.micStream = await navigator.mediaDevices.getUserMedia({
      video: false,
      audio: true,
    });
    console.log(
      'Using ' + this.micStream.getTracks()[0].label + ' as audio input.',
    );
    const audioChunks = aiae.content.mediaStreamToAudioChunks(this.micStream);
    for await (const chunk of audioChunks) {
      this.send({
        part: {
          inline_data: {
            data: aiae.base64.encode(chunk.data),
            mime_type: aiae.content.stringifyMimetype(chunk.metadata.mimetype),
          },
        },
        role: 'user',
        substream_name: 'realtime',
      });
    }
  }

  private micOff() {
    this.send({
      mimetype: 'application/x-state',
      metadata: {mic: 'off'},
    });
    this.mic = false;
    this.micStream?.getTracks().forEach((t) => {
      t.stop();
    });
  }

  private async videoOn(sourceId: String) {
    this.screenOff();
    if (this.video != sourceId) {
      this.videoOff();
    }

    this.video = sourceId;
    this.videoStream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: {exact: sourceId},
        width: {min: 1280, ideal: 1920, max: 1920},
        height: {min: 720, ideal: 1080, max: 1080},
      },
      audio: false,
    });
    this.videoOut.srcObject = this.videoStream;

    this.videoOut.classList.add('visible');

    const videoChunks = aiae.content.mediaStreamToImageChunks(this.videoStream);
    for await (const chunk of videoChunks) {
      this.send({
        part: {
          inline_data: {
            data: aiae.base64.encode(chunk.data),
            mime_type: aiae.content.stringifyMimetype(chunk.metadata.mimetype),
          },
        },
        role: 'user',
        substream_name: 'realtime',
      });
    }
  }

  private videoOff() {
    if (!this.video) return;
    this.video = '';
    this.videoOut.classList.remove('visible');
    this.videoStream?.getTracks().forEach((t) => {
      t.stop();
    });
  }

  private async screenOn() {
    this.videoOff();

    this.screen = true;
    this.screenStream = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: false,
    });
    this.screenOut.srcObject = this.screenStream;

    this.screenOut.classList.add('visible');

    const screenChunks = aiae.content.mediaStreamToImageChunks(
      this.screenStream,
    );
    for await (const chunk of screenChunks) {
      this.send({
        part: {
          inline_data: {
            data: aiae.base64.encode(chunk.data),
            mime_type: aiae.content.stringifyMimetype(chunk.metadata.mimetype),
          },
        },
        role: 'user',
        substream_name: 'realtime',
      });
    }
  }

  private screenOff() {
    this.screen = false;
    this.screenOut.classList.remove('visible');
    this.screenStream?.getTracks().forEach((t) => {
      t.stop();
    });
  }

  private transcriptOn() {
    this.showTranscript = true;
    this.transcriptBox.classList.add('visible');
  }

  private transcriptOff() {
    this.showTranscript = false;
    this.transcriptBox.classList.remove('visible');
  }

  private reset() {
    this.send({
      mimetype: 'application/x-command',
      metadata: {command: 'reset'},
    });
    this.initializeAudioOut();
    this.transcript = '';
  }

  private updateChattiness(event: Event) {
    const target = event.target as HTMLInputElement;
    this.chattiness = parseFloat(target.value);
  }

  private sendChattiness() {
    this.send({
      mimetype: 'application/x-config',
      metadata: {'chattiness': this.chattiness},
    });
    this.initializeAudioOut();
    this.transcript = '';
  }

  protected override render() {
    const micCb = () => {
      if (this.mic) this.micOff();
      else void this.micOn();
    };
    const screenCb = () => {
      if (this.screen) this.screenOff();
      else void this.screenOn();
    };
    const transcriptCb = () => {
      if (this.showTranscript) this.transcriptOff();
      else void this.transcriptOn();
    };

    let videoButtons = '';
    for (const sourceId in this.videoSources) {
      const videoCb = () => {
        let enableVideo = this.video != sourceId;
        this.videoOff();
        if (enableVideo) this.videoOn(sourceId);
      };

      videoButtons = html`${videoButtons}
        <md-icon-button
          @click="${videoCb}"
          toggle
          ?selected="${this.video == sourceId}">
          <md-icon>videocam_off</md-icon>
          <md-icon slot="selected">videocam</md-icon>
        </md-icon-button>`;
    }

    return html`
      <div class="container">
        <div class="main">
          <audio class="audio-out"></audio>
          <video autoplay muted class="video-out"></video>
          <video autoplay muted class="screen-out"></video>
          <div class="transcript-box">${this.transcript}</div>
        </div>
        <div class="controls">
          <div class="controls-left">
            <md-icon-button class="reset-button" @click="${this.reset}">
              <md-icon>autorenew</md-icon>
            </md-icon-button>
          </div>
          <div class="controls-center">
            <md-icon-button @click="${micCb}" toggle ?selected="${this.mic}">
              <md-icon>mic_off</md-icon>
              <md-icon slot="selected">mic</md-icon>
            </md-icon-button>
            ${videoButtons}
            <md-icon-button
              @click="${screenCb}"
              toggle
              ?selected="${this.screen}">
              <md-icon>devices_off</md-icon>
              <md-icon slot="selected">devices</md-icon>
            </md-icon-button>
          </div>
          <div class="controls-right">
            <div class="chattiness-container">
              <label for="chattinessSlider" class="slider-label"
                >CHATTINESS: ${this.chattiness.toFixed(1)}</label
              >
              <input
                type="range"
                id="chattinessSlider"
                name="chattiness"
                min="0"
                max="1"
                step="0.1"
                value="${this.chattiness}"
                @input="${this.updateChattiness}"
                @change="${this.sendChattiness}" />
            </div>
            <md-icon-button
              @click="${transcriptCb}"
              toggle
              ?selected="${this.showTranscript}">
              <md-icon>subtitles_off</md-icon>
              <md-icon slot="selected">subtitles</md-icon>
            </md-icon-button>
          </div>
        </div>
      </div>
    `;
  }

  static override styles = css`
    .container {
      height: 100vh;
    }
    .main {
      height: 95%;
      display: flex;
      justify-content: center;
      align-items: center;
      position: relative;
      flex-wrap: wrap;
    }
    .audio-out,
    .video-out,
    .screen-out,
    .transcript-box {
      display: none;
    }
    .video-out.visible,
    .screen-out.visible,
    .transcript-box.visible {
      display: flex;
    }
    .video-out,
    .screen-out {
      height: 98%;
      width: 98%;
      object-fit: contain;
    }
    .transcript-box {
      position: absolute;
      background-color: rgba(20, 20, 20, 0.6);
      bottom: 5%;
      font-size: x-large;
      max-width: 80%;
      color: white;
    }
    .controls {
      height: 5%;
      width: 100%;
      display: flex;
      align-items: center;
      background-color: lightgray;
    }
    .controls-left {
      display: flex;
      align-items: center;
    }
    .controls-center {
      display: flex;
      align-items: center;
      position: absolute;
      left: 50%;
      transform: translateX(-50%);
    }
    .controls-right {
      display: flex;
      align-items: center;
      margin-left: auto;
    }
    .chattiness-container {
      display: flex;
      align-items: center;
      gap: 0.5vw;
      color: #49454f;
      font-weight: bold;
      font-family: monospace;
      margin-right: 3vw;
    }
  `;
}

function main() {
  const el = document.createElement('live-commentator');
  document.body.appendChild(el);
}

main();
