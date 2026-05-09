import { useEffect, useRef } from "react";

const LETTERS_AND_SYMBOLS = [
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
  "I",
  "J",
  "K",
  "L",
  "M",
  "N",
  "O",
  "P",
  "Q",
  "R",
  "S",
  "T",
  "U",
  "V",
  "W",
  "X",
  "Y",
  "Z",
  "!",
  "@",
  "#",
  "$",
  "&",
  "*",
  "(",
  ")",
  "-",
  "_",
  "+",
  "=",
  "/",
  "[",
  "]",
  "{",
  "}",
  ";",
  ":",
  "<",
  ">",
  ",",
  "0",
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
];

const FONT_SIZE = 16;
const CHAR_WIDTH = 10;
const CHAR_HEIGHT = 20;
const DEFAULT_GLITCH_COLORS = ["#2b4539", "#61dca3", "#61b3dc"];

function randomItem(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function hexToRgb(hex) {
  const normalized = hex.replace(
    /^#?([a-f\d])([a-f\d])([a-f\d])$/i,
    (_match, r, g, b) => `${r}${r}${g}${g}${b}${b}`
  );
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(normalized);

  return result
    ? {
        r: Number.parseInt(result[1], 16),
        g: Number.parseInt(result[2], 16),
        b: Number.parseInt(result[3], 16),
      }
    : null;
}

function interpolateColor(start, end, factor) {
  return `rgb(${Math.round(start.r + (end.r - start.r) * factor)}, ${Math.round(
    start.g + (end.g - start.g) * factor
  )}, ${Math.round(start.b + (end.b - start.b) * factor)})`;
}

export default function LetterGlitchBackground({
  glitchColors = DEFAULT_GLITCH_COLORS,
  glitchSpeed = 60,
  smooth = true,
  centerVignette = false,
  outerVignette = true,
  backgroundColor = "#05060a",
}) {
  const canvasRef = useRef(null);
  const animationRef = useRef(0);
  const contextRef = useRef(null);
  const lettersRef = useRef([]);
  const gridRef = useRef({ columns: 0, rows: 0 });
  const lastGlitchTimeRef = useRef(Date.now());

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return undefined;

    contextRef.current = context;

    function getRandomChar() {
      return randomItem(LETTERS_AND_SYMBOLS);
    }

    function getRandomColor() {
      return randomItem(glitchColors);
    }

    function initializeLetters(columns, rows) {
      gridRef.current = { columns, rows };
      lettersRef.current = Array.from({ length: columns * rows }, () => ({
        char: getRandomChar(),
        color: getRandomColor(),
        targetColor: getRandomColor(),
        colorProgress: 1,
      }));
    }

    function drawLetters() {
      const ctx = contextRef.current;
      if (!ctx || !lettersRef.current.length) return;

      const { width, height } = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, width, height);
      ctx.font = `${FONT_SIZE}px monospace`;
      ctx.textBaseline = "top";

      lettersRef.current.forEach((letter, index) => {
        const x = (index % gridRef.current.columns) * CHAR_WIDTH;
        const y = Math.floor(index / gridRef.current.columns) * CHAR_HEIGHT;
        ctx.fillStyle = letter.color;
        ctx.fillText(letter.char, x, y);
      });
    }

    function resizeCanvas() {
      const parent = canvas.parentElement;
      if (!parent) return;

      const dpr = window.devicePixelRatio || 1;
      const rect = parent.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);

      initializeLetters(Math.ceil(rect.width / CHAR_WIDTH), Math.ceil(rect.height / CHAR_HEIGHT));
      drawLetters();
    }

    function updateLetters() {
      const updateCount = Math.max(1, Math.floor(lettersRef.current.length * 0.05));

      for (let i = 0; i < updateCount; i += 1) {
        const index = Math.floor(Math.random() * lettersRef.current.length);
        const letter = lettersRef.current[index];
        if (!letter) continue;

        letter.char = getRandomChar();
        letter.targetColor = getRandomColor();

        if (smooth) {
          letter.colorProgress = 0;
        } else {
          letter.color = letter.targetColor;
          letter.colorProgress = 1;
        }
      }
    }

    function handleSmoothTransitions() {
      let needsRedraw = false;

      lettersRef.current.forEach((letter) => {
        if (letter.colorProgress >= 1) return;

        letter.colorProgress = Math.min(1, letter.colorProgress + 0.05);
        const startRgb = hexToRgb(letter.color);
        const endRgb = hexToRgb(letter.targetColor);

        if (startRgb && endRgb) {
          letter.color = interpolateColor(startRgb, endRgb, letter.colorProgress);
          needsRedraw = true;
        }
      });

      if (needsRedraw) {
        drawLetters();
      }
    }

    function animate() {
      const now = Date.now();

      if (now - lastGlitchTimeRef.current >= glitchSpeed) {
        updateLetters();
        drawLetters();
        lastGlitchTimeRef.current = now;
      }

      if (smooth) {
        handleSmoothTransitions();
      }

      animationRef.current = requestAnimationFrame(animate);
    }

    let resizeTimeout = 0;

    function handleResize() {
      window.clearTimeout(resizeTimeout);
      resizeTimeout = window.setTimeout(() => {
        cancelAnimationFrame(animationRef.current);
        resizeCanvas();
        animate();
      }, 100);
    }

    resizeCanvas();
    animate();
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(animationRef.current);
      window.clearTimeout(resizeTimeout);
      window.removeEventListener("resize", handleResize);
    };
  }, [glitchColors, glitchSpeed, smooth]);

  return (
    <div className="letter-glitch-shell" style={{ backgroundColor }} aria-hidden="true">
      <canvas ref={canvasRef} className="letter-glitch-canvas" />
      {outerVignette && <div className="letter-glitch-outer-vignette" />}
      {centerVignette && <div className="letter-glitch-center-vignette" />}
    </div>
  );
}
