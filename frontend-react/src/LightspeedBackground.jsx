import { useEffect, useRef, useState } from "react";

const fragmentShader = `#version 300 es
precision highp float;

out vec4 O;

uniform float time;
uniform vec2 resolution;
uniform float intensity;
uniform float particleCount;
uniform vec3 colorShift;

#define FC gl_FragCoord.xy
#define R resolution
#define T time

float rnd(float a) {
  vec2 p = fract(a * vec2(12.9898, 78.233));
  p += dot(p, p * 345.0);
  return fract(p.x * p.y);
}

vec3 hue(float a) {
  return colorShift * (0.6 + 0.6 * cos(6.3 * a + vec3(0.0, 83.0, 21.0)));
}

vec3 pattern(vec2 uv) {
  vec3 col = vec3(0.0);
  for (float i = 0.0; i < particleCount; i++) {
    float a = rnd(i);
    vec2 n = vec2(a, fract(a * 34.56));
    vec2 p = sin(n * (T + 7.0) + T * 0.5);
    float d = max(dot(uv - p, uv - p), 0.001);
    col += (intensity * 0.00125) / d * hue(dot(uv, uv) + i * 0.125 + T);
  }
  return col;
}

void main(void) {
  vec2 uv = (FC - 0.5 * R) / min(R.x, R.y);
  vec3 col = vec3(0.0);
  float s = 2.4;
  float a = atan(uv.x, uv.y);
  float b = length(uv);
  uv = vec2(a * 5.0 / 6.28318, 0.05 / tan(b) + T);
  uv = fract(uv) - 0.5;
  col += pattern(uv * s);
  O = vec4(col, 1.0);
}`;

const vertexShader = `#version 300 es
precision highp float;

in vec2 position;

void main() {
  gl_Position = vec4(position, 0.0, 1.0);
}`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader) || "Shader compile error";
    gl.deleteShader(shader);
    throw new Error(info);
  }

  return shader;
}

function createProgram(gl) {
  const vertex = compileShader(gl, gl.VERTEX_SHADER, vertexShader);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentShader);
  const program = gl.createProgram();

  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);

  gl.deleteShader(vertex);
  gl.deleteShader(fragment);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program) || "Program link error";
    gl.deleteProgram(program);
    throw new Error(info);
  }

  return program;
}

export default function LightspeedBackground({
  speed = 0.65,
  intensity = 0.7,
  particleCount = 20,
  color = [0.72, 0.88, 1],
  quality = "medium",
}) {
  const canvasRef = useRef(null);
  const [webglOk, setWebglOk] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: "high-performance",
    });

    if (!gl) {
      setWebglOk(false);
      return undefined;
    }

    let frame = 0;
    let lastFrame = 0;
    let program;
    let buffer;
    let observer;

    try {
      program = createProgram(gl);
    } catch (error) {
      console.error(error);
      setWebglOk(false);
      return undefined;
    }

    setWebglOk(true);
    gl.useProgram(program);

    buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, 1, -1, -1, 1, 1, 1, -1]), gl.STATIC_DRAW);

    const position = gl.getAttribLocation(program, "position");
    const uniforms = {
      time: gl.getUniformLocation(program, "time"),
      resolution: gl.getUniformLocation(program, "resolution"),
      intensity: gl.getUniformLocation(program, "intensity"),
      particleCount: gl.getUniformLocation(program, "particleCount"),
      colorShift: gl.getUniformLocation(program, "colorShift"),
    };

    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

    const dprMap = { low: 0.5, medium: 1, high: 1.5 };
    const targetDpr = dprMap[quality] || dprMap.medium;

    const resize = () => {
      const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, targetDpr));
      const width = canvas.clientWidth || window.innerWidth;
      const height = canvas.clientHeight || window.innerHeight;
      canvas.width = Math.max(1, Math.floor(width * dpr));
      canvas.height = Math.max(1, Math.floor(height * dpr));
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(uniforms.resolution, canvas.width, canvas.height);
    };

    observer = new ResizeObserver(resize);
    observer.observe(canvas);
    window.addEventListener("resize", resize);
    resize();

    const started = performance.now();
    const loop = (now) => {
      frame = requestAnimationFrame(loop);
      const delta = now - lastFrame;
      if (delta < 1000 / 60) return;

      lastFrame = now - (delta % (1000 / 60));
      gl.useProgram(program);
      gl.uniform1f(uniforms.time, ((now - started) / 1000) * speed);
      gl.uniform1f(uniforms.intensity, intensity);
      gl.uniform1f(uniforms.particleCount, particleCount);
      gl.uniform3f(uniforms.colorShift, color[0], color[1], color[2]);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    };

    frame = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(frame);
      observer?.disconnect();
      window.removeEventListener("resize", resize);
      if (buffer) gl.deleteBuffer(buffer);
      if (program) gl.deleteProgram(program);
    };
  }, [color, intensity, particleCount, quality, speed]);

  return (
    <div className="lightspeed-shell" aria-hidden="true">
      {!webglOk && <div className="lightspeed-fallback" />}
      <canvas ref={canvasRef} className="lightspeed-canvas" />
      <div className="lightspeed-vignette" />
    </div>
  );
}
