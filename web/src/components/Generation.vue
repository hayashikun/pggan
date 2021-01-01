<template>
  <div>
    <h1>PGGAN</h1>
    <button @click="eval">Eval</button>

    <div>
      <label><input v-model="input[0]" type="number" max="1" min="-1" step="0.01"/></label>
      <label><input v-model="input[1]" type="number" max="1" min="-1" step="0.01"/></label>
      <label><input v-model="input[2]" type="number" max="1" min="-1" step="0.01"/></label>
      <label><input v-model="input[3]" type="number" max="1" min="-1" step="0.01"/></label>
    </div>
  </div>
</template>

<script>
import {InferenceSession, Tensor} from 'onnxjs'
import config from "@/assets/config.json";

export default {
  name: "Generation",
  data() {
    const session = new InferenceSession({backendHint: "cpu"})
    return {
      session,
      input: new Float32Array(config.LATENT_VECTOR_SIZE)
    }
  },
  async mounted() {
    await this.session.loadModel("/generator.onnx");
  },
  methods: {
    async eval() {
      const tensor = new Tensor(this.input, 'float32', [1, config.LATENT_VECTOR_SIZE, 1, 1]);
      console.log(this.input, tensor)
      const data = await this.session.run([tensor]);
      console.log(data);
    }
  }
}
</script>

<style scoped>

</style>