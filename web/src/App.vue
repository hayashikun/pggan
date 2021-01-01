<template>
  <div id="app">
    <Generation/>
  </div>
</template>

<script>
import Generation from "@/components/Generation";

export default {
  name: 'App',
  components: {
    Generation
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
