class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.inputSamples = [];
    this.inputPosition = 0;
    this.outputSamples = [];
    this.ratio = sampleRate / 16000;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    for (let index = 0; index < channel.length; index += 1) {
      this.inputSamples.push(channel[index]);
    }

    while (this.inputPosition + 1 < this.inputSamples.length) {
      const leftIndex = Math.floor(this.inputPosition);
      const fraction = this.inputPosition - leftIndex;
      const left = this.inputSamples[leftIndex];
      const right = this.inputSamples[leftIndex + 1];
      const sample = Math.max(-1, Math.min(1, left + (right - left) * fraction));
      this.outputSamples.push(sample < 0 ? sample * 32768 : sample * 32767);
      this.inputPosition += this.ratio;
    }

    const consumed = Math.floor(this.inputPosition);
    if (consumed > 0) {
      this.inputSamples = this.inputSamples.slice(consumed);
      this.inputPosition -= consumed;
    }

    while (this.outputSamples.length >= 512) {
      const pcm = new Int16Array(512);
      for (let index = 0; index < pcm.length; index += 1) {
        pcm[index] = this.outputSamples[index];
      }
      this.outputSamples.splice(0, pcm.length);
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
