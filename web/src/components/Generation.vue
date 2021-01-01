<template>
  <div>
    <h1>PGGAN</h1>
    <button @click="eval">Eval</button>

    <div style="margin: 12px">
      <label><input v-model="input[0]" type="number" max="1" min="-1" step="0.1"/></label>
      <label><input v-model="input[1]" type="number" max="1" min="-1" step="0.1"/></label>
      <label><input v-model="input[2]" type="number" max="1" min="-1" step="0.1"/></label>
      <label><input v-model="input[3]" type="number" max="1" min="-1" step="0.1"/></label>
      <label><input v-model="input[4]" type="number" max="1" min="-1" step="0.1"/></label>
      <label><input v-model="input[5]" type="number" max="1" min="-1" step="0.1"/></label>
      <label><input v-model="input[6]" type="number" max="1" min="-1" step="0.1"/></label>
      <label><input v-model="input[7]" type="number" max="1" min="-1" step="0.1"/></label>
      <label><input v-model="input[8]" type="number" max="1" min="-1" step="0.1"/></label>
      <label><input v-model="input[9]" type="number" max="1" min="-1" step="0.1"/></label>
      <label><input v-model="input[10]" type="number" max="1" min="-1" step="0.1"/></label>
      <label><input v-model="input[11]" type="number" max="1" min="-1" step="0.1"/></label>
      <label><input v-model="input[12]" type="number" max="1" min="-1" step="0.1"/></label>
      <label><input v-model="input[13]" type="number" max="1" min="-1" step="0.1"/></label>
      <label><input v-model="input[14]" type="number" max="1" min="-1" step="0.1"/></label>
      <label><input v-model="input[15]" type="number" max="1" min="-1" step="0.1"/></label>
    </div>
    <canvas id="hidden-canvas" :height="imageSize + 'px'" :width="imageSize + 'px'" style="display: none"/>
    <canvas id="display-canvas" height="400px" width="400px"/>
  </div>
</template>

<script>
import {InferenceSession, Tensor} from "onnxjs"
import config from "@/assets/config.json";

export default {
  name: "Generation",
  data() {
    const session = new InferenceSession({backendHint: "cpu"})
    return {
      session,
      input: new Float32Array(config.LATENT_VECTOR_SIZE),
      imageSize: Math.pow(2, config.MAX_RESOLUTION)
    }
  },
  async mounted() {
    await this.session.loadModel("/generator.onnx");
  },
  methods: {
    async eval() {
      const inTensor = new Tensor(this.input, "float32", [1, config.LATENT_VECTOR_SIZE, 1, 1]);
      const result = await this.session.run([inTensor]);
      const value = result.values().next().value;
      this.draw(value.data);
    },
    draw(data) {
      const hCanvas = document.getElementById("hidden-canvas");
      const hContext = hCanvas.getContext("2d");
      const img = hContext.createImageData(this.imageSize, this.imageSize);
      const pxLen = this.imageSize * this.imageSize;
      for (let i = 0; i < pxLen; i++) {
        img.data[i * 4] = data[i] * 255;
        img.data[i * 4 + 1] = data[i + pxLen] * 255;
        img.data[i * 4 + 2] = data[i + pxLen * 2] * 255;
        img.data[i * 4 + 3] = 255
      }
      hContext.putImageData(img, 0, 0);

      const dCanvas = document.getElementById("display-canvas");
      const dContext = dCanvas.getContext("2d");
      dContext.save();
      dContext.scale(
          dContext.canvas.width / hContext.canvas.width,
          dContext.canvas.height / hContext.canvas.height
      )
      dContext.drawImage(hCanvas, 0, 0)
      dContext.restore()
    }
  }
}
</script>

<style scoped>

</style>