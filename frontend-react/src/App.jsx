import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, LayoutGroup, motion } from "framer-motion";
import { VercelV0Chat } from "@/components/ui/v0-ai-chat";
import { AuthModal } from "@/components/auth/AuthModal";
import { useAuthSession } from "@/components/auth/AuthProvider";
import { useAuthModal } from "@/hooks/useAuthModal";
import { useAgentStream } from "@/hooks/useAgentStream";
import {
  fetchCurrentProfile,
  listUserProjects,
  profileFromUser,
  renameCloudProject,
  saveProfileAppearanceSettings,
  saveProjectToCloud,
  touchCloudProject,
} from "@/lib/cloudData";
import LetterGlitchBackground from "./LetterGlitchBackground.jsx";
import {
  Bot,
  Blocks,
  Braces,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  Code2,
  Copy,
  Download,
  FileCode2,
  FileText,
  Folder,
  Home,
  History,
  ImagePlus,
  Lightbulb,
  MailPlus,
  Maximize2,
  MessageSquare,
  Mic,
  Monitor,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Palette,
  Paperclip,
  PenLine,
  Play,
  Plus,
  RefreshCw,
  Rocket,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  Sun,
  Terminal,
  GitBranch,
  Link,
  Users,
  X,
} from "lucide-react";

const fileOrder = ["index.html", "styles.css", "script.js"];
const layoutTransition = { duration: 0.44, ease: "easeInOut" };
const frameTransition = { duration: 0.72, ease: [0.76, 0, 0.24, 1] };
const DESIGN_SETTINGS_STORAGE_KEY = "mindi-design-settings-v1";
const DEFAULT_DESIGN_SETTINGS = {
  headline: "Website",
  body: "A sharp web surface generated from your prompt, with focused layout, native controls, and clean production code.",
  cta: "Preview flow",
  accent: "#22c55e",
  background: "#101616",
  font: "Inter",
  radius: 14,
  spacing: 28,
};
const APPEARANCE_SETTINGS_STORAGE_KEY = "mindi-appearance-settings-v1";
const DEFAULT_APPEARANCE_SETTINGS = {
  theme: "dark",
  backgroundEnabled: true,
  chatColorEnabled: true,
  overlay: "#22c55e",
  blur: true,
  applyAll: false,
  backgroundImage: "dusk",
  customBackgroundUrl: "",
};
const APPEARANCE_BACKGROUNDS = [
  {
    id: "mountains",
    label: "Alpine ridgeline",
    url: "https://images.unsplash.com/photo-1501785888041-af3ef285b470?auto=format&fit=crop&w=2400&q=90",
  },
  {
    id: "desert",
    label: "Desert nightfall",
    url: "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?auto=format&fit=crop&w=2400&q=90",
  },
  {
    id: "glacier",
    label: "Glacier summit",
    url: "https://images.unsplash.com/photo-1483728642387-6c3bdd6c93e5?auto=format&fit=crop&w=2400&q=90",
  },
  {
    id: "dusk",
    label: "Dusk granite",
    url: "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=2400&q=90",
  },
  {
    id: "blue-abstract",
    label: "Blue abstract",
    url: "https://images.unsplash.com/photo-1557683316-973673baf926?auto=format&fit=crop&w=2400&q=90",
  },
  {
    id: "orange-abstract",
    label: "Orange abstract",
    url: "https://images.unsplash.com/photo-1618005198919-d3d4b5a92ead?auto=format&fit=crop&w=2400&q=90",
  },
  {
    id: "chrome-abstract",
    label: "Chrome abstract",
    url: "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=2400&q=90",
  },
  {
    id: "waveform",
    label: "Waveform mesh",
    url: "https://images.unsplash.com/photo-1635776062043-223faf322554?auto=format&fit=crop&w=2400&q=90",
  },
];
const APPEARANCE_BACKGROUND_BY_ID = Object.fromEntries(APPEARANCE_BACKGROUNDS.map((background) => [background.id, background]));
const DEFAULT_WEATHER_LOCATION = {
  latitude: 18.5204,
  longitude: 73.8567,
  label: "Pune",
};
const WEATHER_CODE_LABELS = {
  0: "Clear",
  1: "Mostly clear",
  2: "Partly cloudy",
  3: "Cloudy",
  45: "Fog",
  48: "Rime fog",
  51: "Light drizzle",
  53: "Drizzle",
  55: "Heavy drizzle",
  61: "Light rain",
  63: "Rain",
  65: "Heavy rain",
  71: "Light snow",
  73: "Snow",
  75: "Heavy snow",
  80: "Rain showers",
  81: "Rain showers",
  82: "Heavy showers",
  95: "Thunderstorm",
  96: "Thunderstorm",
  99: "Thunderstorm",
};
const WEATHER_ICON_URLS = {
  clear: new URL("../weather_icons_animated&static/animated/day.svg", import.meta.url).href,
  mostlyClear: new URL("../weather_icons_animated&static/animated/cloudy-day-1.svg", import.meta.url).href,
  partlyCloudy: new URL("../weather_icons_animated&static/animated/cloudy-day-2.svg", import.meta.url).href,
  cloudy: new URL("../weather_icons_animated&static/animated/cloudy.svg", import.meta.url).href,
  fog: new URL("../weather_icons_animated&static/animated/weather.svg", import.meta.url).href,
  drizzle: new URL("../weather_icons_animated&static/animated/rainy-2.svg", import.meta.url).href,
  rain: new URL("../weather_icons_animated&static/animated/rainy-4.svg", import.meta.url).href,
  heavyRain: new URL("../weather_icons_animated&static/animated/rainy-6.svg", import.meta.url).href,
  snow: new URL("../weather_icons_animated&static/animated/snowy-4.svg", import.meta.url).href,
  thunder: new URL("../weather_icons_animated&static/animated/thunder.svg", import.meta.url).href,
  fallback: new URL("../weather_icons_animated&static/animated/weather.svg", import.meta.url).href,
};
const userAvatar =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
      <rect width="64" height="64" rx="32" fill="#171717"/>
      <circle cx="32" cy="23" r="12" fill="#f4d7cc"/>
      <path d="M11 62c2.8-15.4 12-24 21-24s18.2 8.6 21 24H11Z" fill="#292d32"/>
      <path d="M20 18c2-8 8-12 17-10 8 1.8 12.6 7.4 12 16-5.6-5.2-12.8-8.1-29-6Z" fill="#111"/>
      <circle cx="50" cy="50" r="9" fill="#22c55e"/>
      <path d="M46 50.4 49 53l5-6" fill="none" stroke="#03120b" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  `);

const starterFiles = {
  "index.html": `<main class="page">
  <section class="hero">
    <nav>
      <strong>Website</strong>
      <button id="contact">Start</button>
    </nav>
    <div class="hero-content">
      <div>
        <h1>Website</h1>
        <p>A sharp web surface generated from your prompt, with focused layout, native controls, and clean production code.</p>
        <button id="primaryAction">Preview flow</button>
      </div>
      <figure>
        <div class="screen-line wide"></div>
        <div class="screen-line"></div>
        <div class="screen-block"></div>
      </figure>
    </div>
  </section>
</main>`,
  "styles.css": `:root {
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  background: #08090c;
  color: #f6f8fb;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  background:
    linear-gradient(135deg, rgba(129, 231, 255, 0.12), transparent 32%),
    linear-gradient(315deg, rgba(210, 157, 255, 0.14), transparent 38%),
    #08090c;
}

.page { min-height: 100vh; padding: 28px; }
.hero {
  min-height: calc(100vh - 56px);
  display: grid;
  grid-template-rows: auto 1fr;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.04);
  backdrop-filter: blur(18px);
}

nav, .hero-content {
  width: min(1120px, 100%);
  margin: 0 auto;
}

nav {
  height: 72px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.hero-content {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 420px;
  gap: 64px;
  align-items: center;
}

h1 {
  margin: 0;
  max-width: 720px;
  font-size: clamp(42px, 7vw, 86px);
  line-height: 0.95;
}

p {
  max-width: 560px;
  color: #b8beca;
  font-size: 18px;
  line-height: 1.7;
}

button {
  border: 0;
  background: #f6f8fb;
  color: #07080b;
  padding: 13px 18px;
  font-weight: 700;
  cursor: pointer;
}

figure {
  min-height: 460px;
  margin: 0;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.08);
  display: grid;
  gap: 16px;
  align-content: center;
  padding: 28px;
}

.screen-line, .screen-block {
  height: 82px;
  background: rgba(255, 255, 255, 0.13);
}

.screen-line.wide { width: 100%; }
.screen-line { width: 76%; }

@media (max-width: 840px) {
  .hero-content { grid-template-columns: 1fr; gap: 32px; padding: 32px 0; }
  figure { min-height: 260px; }
}`,
  "script.js": `const action = document.querySelector("#primaryAction");

action?.addEventListener("click", () => {
  action.textContent = "Running";
  document.body.classList.add("active");
});`,
};

const workspaceTemplates = {
  "header.tsx": `import { Button } from "@/components/ui/button";

export function Header() {
  return (
    <header className="sticky top-0 z-40 flex items-center justify-between border-b border-white/10 bg-black/70 px-6 py-4 backdrop-blur">
      <strong>MINDIGENOUS</strong>
      <nav className="flex items-center gap-5 text-sm text-white/70">
        <a href="#features">Features</a>
        <a href="#pricing">Pricing</a>
        <Button>Start building</Button>
      </nav>
    </header>
  );
}`,
  "banner.tsx": `import React from "react";

const slides = [
  {
    id: 1,
    title: "AI websites ready for production",
    eyebrow: "MINDIGENOUS builder",
  },
];

export function Banner() {
  return (
    <section className="grid min-h-screen place-items-center bg-neutral-950 text-white">
      <div className="mx-auto max-w-4xl px-6 text-center">
        <p>{slides[0].eyebrow}</p>
        <h1>{slides[0].title}</h1>
      </div>
    </section>
  );
}`,
  "codeeditor.tsx": `export function CodeEditor({ value, onChange }) {
  return (
    <textarea
      value={value}
      onChange={(event) => onChange(event.target.value)}
      spellCheck={false}
      className="h-full w-full resize-none bg-transparent font-mono text-sm outline-none"
    />
  );
}`,
  "footer.tsx": `export function Footer() {
  return (
    <footer className="border-t border-white/10 px-6 py-8 text-sm text-white/50">
      Built with MINDIGENOUS.
    </footer>
  );
}`,
  "global.css": `@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  color-scheme: dark;
  background: #050706;
  color: #f5f7f5;
}`,
  "product.js": `export const product = {
  name: "MINDIGENOUS",
  mode: "AI web builder",
  deployTarget: "AMD cloud",
};`,
  "router.tsx": `import { createBrowserRouter } from "react-router-dom";

export const router = createBrowserRouter([
  { path: "/", element: <main id="app" /> },
]);`,
  "package.json": `{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@vitejs/plugin-react": "latest",
    "framer-motion": "latest",
    "lucide-react": "latest",
    "react": "latest",
    "react-dom": "latest"
  }
}`,
  "README.md": `# MINDIGENOUS

AI web builder workspace with prompt, code, preview, and deploy handoff.`,
  ".gitignore": `node_modules
dist
.env
.DS_Store`,
};

function normalizeDesignSettings(settings = {}) {
  const radius = Number(settings.radius);
  const spacing = Number(settings.spacing);

  return {
    ...DEFAULT_DESIGN_SETTINGS,
    ...settings,
    radius: Number.isFinite(radius) ? radius : DEFAULT_DESIGN_SETTINGS.radius,
    spacing: Number.isFinite(spacing) ? spacing : DEFAULT_DESIGN_SETTINGS.spacing,
  };
}

function readStoredDesignSettings() {
  if (typeof window === "undefined") return DEFAULT_DESIGN_SETTINGS;

  try {
    const stored = window.localStorage.getItem(DESIGN_SETTINGS_STORAGE_KEY);
    return stored ? normalizeDesignSettings(JSON.parse(stored)) : DEFAULT_DESIGN_SETTINGS;
  } catch {
    return DEFAULT_DESIGN_SETTINGS;
  }
}

function normalizeAppearanceSettings(settings = {}) {
  return { ...DEFAULT_APPEARANCE_SETTINGS, ...settings };
}

function appearanceSettingsStorageKeys(user) {
  const userKey = user?.id || user?.email;
  return userKey
    ? [`${APPEARANCE_SETTINGS_STORAGE_KEY}:${userKey}`, APPEARANCE_SETTINGS_STORAGE_KEY]
    : [APPEARANCE_SETTINGS_STORAGE_KEY];
}

function readStoredAppearanceSettings(user) {
  if (typeof window === "undefined") return DEFAULT_APPEARANCE_SETTINGS;

  try {
    for (const key of appearanceSettingsStorageKeys(user)) {
      const stored = window.localStorage.getItem(key);
      if (stored) return normalizeAppearanceSettings(JSON.parse(stored));
    }
    return DEFAULT_APPEARANCE_SETTINGS;
  } catch {
    return DEFAULT_APPEARANCE_SETTINGS;
  }
}

function writeStoredAppearanceSettings(settings, user) {
  if (typeof window === "undefined") return;

  const payload = JSON.stringify(normalizeAppearanceSettings(settings));
  const keys = new Set(appearanceSettingsStorageKeys(user));
  keys.add(APPEARANCE_SETTINGS_STORAGE_KEY);
  keys.forEach((key) => window.localStorage.setItem(key, payload));
}

function hexToRgbParts(value) {
  const match = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(value || "");
  if (!match) return "34 197 94";
  return `${parseInt(match[1], 16)} ${parseInt(match[2], 16)} ${parseInt(match[3], 16)}`;
}

function resolveAppearanceTheme(theme) {
  if (theme === "light" || theme === "dark") return theme;
  if (typeof window === "undefined" || !window.matchMedia) return "dark";
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function applyAppearanceSettings(settings = DEFAULT_APPEARANCE_SETTINGS) {
  if (typeof document === "undefined") return;

  const normalized = normalizeAppearanceSettings(settings);
  const resolvedTheme = resolveAppearanceTheme(normalized.theme);
  const background = APPEARANCE_BACKGROUND_BY_ID[normalized.backgroundImage] ?? APPEARANCE_BACKGROUND_BY_ID[DEFAULT_APPEARANCE_SETTINGS.backgroundImage];
  const backgroundUrl = normalized.backgroundImage === "custom" && normalized.customBackgroundUrl ? normalized.customBackgroundUrl : background.url;
  const root = document.documentElement;
  root.dataset.appearanceTheme = resolvedTheme;
  root.style.setProperty("--appearance-overlay", normalized.overlay);
  root.style.setProperty("--appearance-overlay-rgb", hexToRgbParts(normalized.overlay));
  root.style.setProperty("--appearance-text", resolvedTheme === "light" ? "#070809" : "#ffffff");
  root.style.setProperty("--appearance-muted-text", resolvedTheme === "light" ? "#4b5563" : "#aab4b1");
  root.style.setProperty("--appearance-background-image", normalized.backgroundEnabled ? `url("${backgroundUrl}")` : "none");
  root.style.setProperty("--appearance-background-opacity", normalized.backgroundEnabled ? "1" : "0");
  root.style.setProperty("--appearance-glitch-opacity", normalized.backgroundEnabled ? "0.22" : "1");
  root.style.setProperty("--appearance-bg-filter", normalized.blur ? "blur(0px)" : "none");
}

function weatherCodeLabel(code) {
  return WEATHER_CODE_LABELS[code] ?? "Live weather";
}

function weatherIconForCode(code) {
  if (code === 0) return WEATHER_ICON_URLS.clear;
  if (code === 1) return WEATHER_ICON_URLS.mostlyClear;
  if (code === 2) return WEATHER_ICON_URLS.partlyCloudy;
  if (code === 3) return WEATHER_ICON_URLS.cloudy;
  if (code === 45 || code === 48) return WEATHER_ICON_URLS.fog;
  if ([51, 53, 55].includes(code)) return WEATHER_ICON_URLS.drizzle;
  if ([61, 63, 80, 81].includes(code)) return WEATHER_ICON_URLS.rain;
  if ([65, 82].includes(code)) return WEATHER_ICON_URLS.heavyRain;
  if ([71, 73, 75].includes(code)) return WEATHER_ICON_URLS.snow;
  if ([95, 96, 99].includes(code)) return WEATHER_ICON_URLS.thunder;
  return WEATHER_ICON_URLS.fallback;
}

function WeatherIcon({ code, className = "" }) {
  return (
    <img
      className={`weather-icon ${className}`}
      src={weatherIconForCode(code)}
      alt=""
      aria-hidden="true"
      draggable="false"
    />
  );
}

function formatProjectTime(value) {
  if (!value) return "Recently opened";

  const then = new Date(value).getTime();
  if (!Number.isFinite(then)) return "Recently opened";

  const seconds = Math.max(1, Math.floor((Date.now() - then) / 1000));
  if (seconds < 60) return "Just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hr ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days} day${days === 1 ? "" : "s"} ago`;
  const months = Math.floor(days / 30);
  return `${months} mo ago`;
}

function displayNameFromProfile(profile, user) {
  return (
    profile?.full_name ||
    user?.user_metadata?.full_name ||
    user?.user_metadata?.name ||
    [user?.user_metadata?.first_name, user?.user_metadata?.last_name].filter(Boolean).join(" ").trim() ||
    user?.email?.split("@")[0] ||
    "Sign in"
  );
}

function avatarFromProfile(profile, user) {
  return profile?.avatar_url || user?.user_metadata?.avatar_url || user?.user_metadata?.picture || userAvatar;
}

function useLocalWeather() {
  const [weather, setWeather] = useState({
    location: DEFAULT_WEATHER_LOCATION.label,
    temperature: null,
    condition: "Detecting location",
    code: null,
    high: null,
    low: null,
    humidity: null,
    wind: null,
    forecast: [],
    source: "loading",
  });

  useEffect(() => {
    let cancelled = false;

    async function fetchWeather(location, source) {
      try {
        const params = new URLSearchParams({
          latitude: String(location.latitude),
          longitude: String(location.longitude),
          current: "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
          daily: "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
          timezone: "auto",
          forecast_days: "5",
        });
        const [forecastResponse, reverseResponse] = await Promise.all([
          fetch(`https://api.open-meteo.com/v1/forecast?${params.toString()}`),
          fetch(`https://geocoding-api.open-meteo.com/v1/reverse?latitude=${location.latitude}&longitude=${location.longitude}&count=1&language=en&format=json`).catch(() => null),
        ]);

        if (!forecastResponse.ok) throw new Error("Weather request failed");
        const forecast = await forecastResponse.json();
        const reverse = reverseResponse?.ok ? await reverseResponse.json() : null;
        const resolvedName = reverse?.results?.[0]?.name ?? location.label;

        if (cancelled) return;
        setWeather({
          location: resolvedName,
          temperature: Math.round(forecast.current?.temperature_2m ?? 0),
          condition: weatherCodeLabel(forecast.current?.weather_code),
          code: forecast.current?.weather_code ?? null,
          high: Math.round(forecast.daily?.temperature_2m_max?.[0] ?? 0),
          low: Math.round(forecast.daily?.temperature_2m_min?.[0] ?? 0),
          humidity: Math.round(forecast.current?.relative_humidity_2m ?? 0),
          wind: Math.round(forecast.current?.wind_speed_10m ?? 0),
          forecast: (forecast.daily?.time ?? []).map((date, index) => ({
            date,
            condition: weatherCodeLabel(forecast.daily?.weather_code?.[index]),
            code: forecast.daily?.weather_code?.[index] ?? null,
            high: Math.round(forecast.daily?.temperature_2m_max?.[index] ?? 0),
            low: Math.round(forecast.daily?.temperature_2m_min?.[index] ?? 0),
            precipitation: Math.round(forecast.daily?.precipitation_probability_max?.[index] ?? 0),
          })),
          source,
        });
      } catch {
        if (cancelled) return;
        setWeather((current) => ({
          ...current,
          temperature: null,
          condition: "Weather unavailable",
          code: null,
          forecast: [],
          source: "unavailable",
        }));
      }
    }

    async function useFallback() {
      try {
        const response = await fetch("https://ipapi.co/json/");
        if (!response.ok) throw new Error("Location lookup failed");
        const location = await response.json();
        const latitude = Number(location.latitude);
        const longitude = Number(location.longitude);

        if (Number.isFinite(latitude) && Number.isFinite(longitude)) {
          fetchWeather(
            {
              latitude,
              longitude,
              label: location.city || DEFAULT_WEATHER_LOCATION.label,
            },
            "network"
          );
          return;
        }
      } catch {
        // Fall through to the project default when browser and network location are unavailable.
      }

      fetchWeather(DEFAULT_WEATHER_LOCATION, "default");
    }

    if (!navigator.geolocation) {
      useFallback();
      return () => {
        cancelled = true;
      };
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        fetchWeather(
          {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            label: DEFAULT_WEATHER_LOCATION.label,
          },
          "device"
        );
      },
      useFallback,
      {
        enableHighAccuracy: false,
        maximumAge: 10 * 60 * 1000,
        timeout: 8000,
      }
    );

    return () => {
      cancelled = true;
    };
  }, []);

  return weather;
}

function escapeHtmlContent(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function createDesignCssBlock(settings) {
  const next = normalizeDesignSettings(settings);

  return `/* MINDIGENOUS design controls */
:root {
  --accent: ${next.accent};
  --surface: ${next.background};
  --radius: ${next.radius}px;
  --space: ${next.spacing}px;
}

body {
  font-family: ${next.font}, Inter, ui-sans-serif, system-ui, sans-serif;
}

.hero {
  background:
    radial-gradient(circle at 76% 28%, color-mix(in srgb, var(--accent), transparent 72%), transparent 30%),
    var(--surface);
  padding: var(--space);
}

button,
figure,
.screen-block {
  border-radius: var(--radius);
}

#primaryAction {
  background: var(--accent);
  color: #04110a;
}
/* end MINDIGENOUS design controls */`;
}

function applyDesignToFiles(files, settings) {
  const next = normalizeDesignSettings(settings);
  const headline = escapeHtmlContent(next.headline);
  const body = escapeHtmlContent(next.body);
  const cta = escapeHtmlContent(next.cta);
  const html = files["index.html"] ?? "";
  const css = files["styles.css"] ?? "";
  const designBlock = createDesignCssBlock(next);
  const boundedDesignBlock = /\/\* MINDIGENOUS design controls \*\/[\s\S]*?\/\* end MINDIGENOUS design controls \*\//;
  const legacyDesignBlock = /\/\* MINDIGENOUS design controls \*\/[\s\S]*$/;

  const nextHtml = html
    .replace(/<strong>[\s\S]*?<\/strong>/, `<strong>${headline}</strong>`)
    .replace(/<h1>[\s\S]*?<\/h1>/, `<h1>${headline}</h1>`)
    .replace(/<p>[\s\S]*?<\/p>/, `<p>${body}</p>`)
    .replace(/<button id="primaryAction"[^>]*>[\s\S]*?<\/button>/, `<button id="primaryAction">${cta}</button>`);
  const nextCss = boundedDesignBlock.test(css)
    ? css.replace(boundedDesignBlock, designBlock)
    : legacyDesignBlock.test(css)
      ? css.replace(legacyDesignBlock, designBlock)
      : `${css.trimEnd()}\n\n${designBlock}`;

  return { ...files, "index.html": nextHtml, "styles.css": nextCss };
}

function templateForFile(file) {
  return workspaceTemplates[file] ?? `// ${file}
// New file created in the MINDIGENOUS workspace.
`;
}

function fileKind(file) {
  if (file.endsWith(".tsx") || file.endsWith(".jsx")) return "react";
  if (file.endsWith(".ts")) return "ts";
  if (file.endsWith(".js")) return "js";
  if (file.endsWith(".html")) return "html";
  if (file.endsWith(".css")) return "css";
  if (file.endsWith(".json")) return "json";
  if (file.endsWith(".md")) return "md";
  if (file === ".gitignore" || file.endsWith(".git")) return "git";
  return "file";
}

function FileIcon({ file, kind }) {
  const resolvedKind = kind ?? fileKind(file);
  const shortLabels = {
    ts: "TS",
    js: "JS",
    html: "H",
    css: "#",
    json: "{}",
    md: "M",
    git: "G",
  };

  return (
    <span className={`file-icon ${resolvedKind}`} aria-hidden="true">
      {resolvedKind === "react" ? (
        <svg viewBox="0 0 24 24" role="img">
          <circle cx="12" cy="12" r="2.4" />
          <ellipse cx="12" cy="12" rx="9" ry="3.5" />
          <ellipse cx="12" cy="12" rx="9" ry="3.5" transform="rotate(60 12 12)" />
          <ellipse cx="12" cy="12" rx="9" ry="3.5" transform="rotate(120 12 12)" />
        </svg>
      ) : shortLabels[resolvedKind] ? (
        shortLabels[resolvedKind]
      ) : (
        <FileCode2 size={12} />
      )}
    </span>
  );
  const labels = {
    react: "⚛",
    ts: "TS",
    js: "JS",
    html: "H",
    css: "#",
    json: "{}",
    md: "M",
    git: "◆",
    file: "•",
  };

  return (
    <span className={`file-icon ${resolvedKind}`} aria-hidden="true">
      {labels[resolvedKind] ?? labels.file}
    </span>
  );
}

function stashMatches(source, pattern, className) {
  const stash = [];
  const marked = source.replace(pattern, (match) => {
    const encodedIndex = String(stash.length)
      .split("")
      .map((digit) => "abcdefghij"[Number(digit)])
      .join("");
    const token = `\uE000${encodedIndex}\uE001`;
    stash.push([token, `<span class="${className}">${match}</span>`]);
    return token;
  });

  return { marked, stash };
}

function restoreStash(source, stash) {
  return stash.reduce((current, [token, html]) => current.replaceAll(token, html), source);
}

function highlightHtmlLine(line) {
  let html = escapeHtml(line);
  const commentResult = stashMatches(html, /&lt;!--.*?--&gt;/g, "token-comment");
  html = commentResult.marked;
  const stringResult = stashMatches(html, /"[^"]*"|'[^']*'/g, "token-string");
  html = stringResult.marked;
  html = html.replace(/\s([a-zA-Z_:][\w:.-]*)(=)/g, ' <span class="token-attr">$1</span>$2');
  html = html.replace(/(&lt;\/?)([a-zA-Z][\w:-]*)/g, '$1<span class="token-tag">$2</span>');
  html = restoreStash(html, stringResult.stash);
  return restoreStash(html, commentResult.stash);
}

function highlightCssLine(line) {
  let html = escapeHtml(line);
  const commentResult = stashMatches(html, /\/\*.*?\*\//g, "token-comment");
  html = commentResult.marked;
  const stringResult = stashMatches(html, /"[^"]*"|'[^']*'/g, "token-string");
  html = stringResult.marked;
  html = html.replace(/([a-z-]+)(\s*:)/gi, '<span class="token-property">$1</span>$2');
  html = html.replace(/(#(?:[0-9a-fA-F]{3}){1,2}|rgba?\([^)]*\)|\b\d+(?:\.\d+)?(?:px|rem|em|vh|vw|%)?\b)/g, '<span class="token-number">$1</span>');
  html = restoreStash(html, stringResult.stash);
  return restoreStash(html, commentResult.stash);
}

function highlightScriptLine(line) {
  let html = escapeHtml(line);
  const commentResult = stashMatches(html, /(\/\/.*|\/\*.*?\*\/)/g, "token-comment");
  html = commentResult.marked;
  const stringResult = stashMatches(html, /`[^`]*`|"[^"]*"|'[^']*'/g, "token-string");
  html = stringResult.marked;
  html = html.replace(/\b(import|from|const|let|var|function|return|if|else|for|while|class|new|export|default|await|async|try|catch|true|false|null|undefined)\b/g, '<span class="token-keyword">$1</span>');
  html = html.replace(/\b([A-Z][A-Za-z0-9_]*|use[A-Z][A-Za-z0-9_]*)\b/g, '<span class="token-type">$1</span>');
  html = html.replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="token-number">$1</span>');
  html = restoreStash(html, stringResult.stash);
  return restoreStash(html, commentResult.stash);
}

function highlightCode(value, file) {
  const kind = fileKind(file);
  const highlighter = kind === "html" ? highlightHtmlLine : kind === "css" ? highlightCssLine : highlightScriptLine;
  return value.split("\n").map((line) => highlighter(line) || " ").join("\n");
}

function isBuildIntent(value) {
  const text = value.toLowerCase();
  const triggers = [
    "build",
    "create",
    "generate",
    "make website",
    "landing page",
    "dashboard",
    "app ui",
    "website",
    "web app",
    "app",
    "ui",
  ];
  const structured = /\n|title:|goal:|sections:|style:|pages:|features:/i.test(value);

  return structured || triggers.some((trigger) => text.includes(trigger));
}

function isVagueBuild(value) {
  const compact = value.trim().toLowerCase();
  return ["build", "create", "generate", "make website", "make app"].includes(compact);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function titleFromPrompt(prompt) {
  const normalized = prompt
    .replace(/^(build|create|generate|make)\s+/i, "")
    .replace(/\b(for|with|using|that|which)\b.*$/i, "")
    .replace(/\s+/g, " ")
    .trim();

  if (!normalized) return "Generated Interface";

  return normalized
    .split(" ")
    .slice(0, 6)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function generateFiles(prompt) {
  const title = titleFromPrompt(prompt);
  const lower = prompt.toLowerCase();
  const safeTitle = escapeHtml(title);

  if (lower.includes("dashboard") || lower.includes("admin") || lower.includes("analytics")) {
    return {
      "index.html": `<main class="dashboard-shell">
  <aside class="rail">
    <strong>${safeTitle}</strong>
    <a class="active">Overview</a>
    <a>Reports</a>
    <a>Settings</a>
  </aside>
  <section class="workspace">
    <header class="topbar">
      <div>
        <p>Live system</p>
        <h1>${safeTitle}</h1>
      </div>
      <button id="refresh">Refresh</button>
    </header>
    <div class="metrics">
      <article><span>Revenue</span><strong>$48.2K</strong></article>
      <article><span>Users</span><strong>12,804</strong></article>
      <article><span>Health</span><strong>98%</strong></article>
    </div>
  </section>
</main>`,
      "styles.css": `:root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; background: #08090f; color: #f7f8fb; }
* { box-sizing: border-box; } body { margin: 0; background: #08090f; }
.dashboard-shell { min-height: 100vh; display: grid; grid-template-columns: 240px 1fr; }
.rail { padding: 28px; background: #10131b; border-right: 1px solid rgba(255,255,255,.08); }
.rail strong { display: block; margin-bottom: 32px; } .rail a { display: block; padding: 12px 0; color: #8e98a8; }
.rail .active { color: #f7f8fb; } .workspace { padding: 32px; }
.topbar { display: flex; justify-content: space-between; gap: 24px; align-items: center; margin-bottom: 24px; }
.topbar p { margin: 0 0 8px; color: #7ce3cc; font-size: 13px; text-transform: uppercase; }
h1 { margin: 0; font-size: clamp(34px, 5vw, 68px); line-height: 1; }
button { border: 0; background: #f7f8fb; color: #08090f; padding: 12px 16px; font-weight: 700; cursor: pointer; }
.metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
article { border: 1px solid rgba(255,255,255,.1); background: rgba(255,255,255,.05); padding: 20px; }
article span { color: #9aa4b3; font-size: 13px; } article strong { display: block; margin-top: 10px; font-size: 32px; }`,
      "script.js": `const refresh = document.querySelector("#refresh");
refresh?.addEventListener("click", () => {
  refresh.textContent = "Synced";
  setTimeout(() => { refresh.textContent = "Refresh"; }, 1400);
});`,
    };
  }

  return {
    "index.html": starterFiles["index.html"]
      .replaceAll("Website", safeTitle)
      .replace("A sharp web surface generated from your prompt", `A sharp web surface generated from your prompt`),
    "styles.css": starterFiles["styles.css"],
    "script.js": starterFiles["script.js"],
  };
}

function composePreview(files) {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>${files["styles.css"]}</style>
</head>
<body>
${files["index.html"]}
<script>${files["script.js"]}</script>
</body>
</html>`;
}

function LiquidLogo({ compact = false }) {
  return (
    <div className={compact ? "liquid-logo compact" : "liquid-logo"} aria-label="MINDIGENOUS">
      <span className="logo-shine-mask">
        <img className="liquid-logo-image" src="/assets/mindigenous-white-logo.png" alt="MINDIGENOUS" />
      </span>
    </div>
  );
}

function WelcomePrompt() {
  return <h1 className="welcome-prompt">What would you like to build?</h1>;
}

function IconButton({ children, className = "", ...props }) {
  return (
    <button className={`icon-button ${className}`} {...props}>
      {children}
    </button>
  );
}

function PromptBox({ value, onChange, onSubmit, busy, placement = "center", frameRef, onPromptOptionsChange }) {
  return (
    <motion.div
      ref={frameRef}
      layout
      layoutId="prompt-box"
      className={`prompt-box ${placement === "top" ? "top" : "center"}`}
      transition={layoutTransition}
      role="form"
    >
      <VercelV0Chat
        value={value}
        onValueChange={onChange}
        onSubmit={onSubmit}
        busy={busy}
        placement={placement}
        showHeading={false}
        showActions
        onMoreMenuChange={onPromptOptionsChange}
      />
    </motion.div>
  );
}

function ChatMode({
  prompt,
  setPrompt,
  onSubmit,
  busy,
  promptFrameRef,
  onNotify,
  user,
  profile,
  projects,
  cloudLoading,
  onOpenProject,
  onOpenUpgrade,
}) {
  const [appearanceModalOpen, setAppearanceModalOpen] = useState(false);
  const [homeSettingsOpen, setHomeSettingsOpen] = useState(false);
  const [homeSidebarCollapsed, setHomeSidebarCollapsed] = useState(false);
  const [promptOptionsOpen, setPromptOptionsOpen] = useState(false);
  const { openAuth } = useAuthModal();

  function handleHomeSuggestion(nextPrompt) {
    setPrompt(nextPrompt);
    window.requestAnimationFrame(() => {
      promptFrameRef.current?.querySelector("textarea")?.focus();
    });
  }

  function handleNewChat() {
    setPrompt("");
    onNotify?.("New chat ready");
    window.requestAnimationFrame(() => {
      promptFrameRef.current?.querySelector("textarea")?.focus();
    });
  }

  return (
    <section className="chat-mode" aria-label="MINDI chat">
      <HomeSidebar
        collapsed={homeSidebarCollapsed}
        onToggleCollapsed={() => setHomeSidebarCollapsed((value) => !value)}
        onNewChat={handleNewChat}
        onSelect={handleHomeSuggestion}
        onOpenProject={onOpenProject}
        onOpenUpgrade={onOpenUpgrade}
        onOpenAuth={() => openAuth("signin")}
        user={user}
        profile={profile}
        projects={projects}
        cloudLoading={cloudLoading}
      />
      <div className="home-top-actions" aria-label="Home settings">
        <button type="button" className="home-appearance-button" onClick={() => setAppearanceModalOpen(true)}>
          <Palette size={14} />
          Appearance
        </button>
        <IconButton className="home-settings-button" onClick={() => setHomeSettingsOpen(true)} aria-label="Open settings">
          <Settings size={17} />
        </IconButton>
      </div>
      <div className="chat-center">
        <div className="home-brand">
          <LiquidLogo />
        </div>
        <WelcomePrompt />
        <PromptBox
          value={prompt}
          onChange={setPrompt}
          onSubmit={onSubmit}
          busy={busy}
          frameRef={promptFrameRef}
          onPromptOptionsChange={setPromptOptionsOpen}
        />
        <HomeBelowPrompt
          onSelect={handleHomeSuggestion}
          projects={projects}
          cloudLoading={cloudLoading}
          onOpenProject={onOpenProject}
          promptOptionsOpen={promptOptionsOpen}
        />
      </div>
      <AnimatePresence>
        {appearanceModalOpen && (
          <SettingsModal
            onClose={() => setAppearanceModalOpen(false)}
            onNotify={onNotify}
            user={user}
          />
        )}
      </AnimatePresence>
      <AnimatePresence>
        {homeSettingsOpen && (
          <HomeSettingsModal
            onClose={() => setHomeSettingsOpen(false)}
            onNotify={onNotify}
            onOpenAuth={() => openAuth("signin")}
            onOpenUpgrade={onOpenUpgrade}
            user={user}
            profile={profile}
          />
        )}
      </AnimatePresence>
    </section>
  );
}

function HomeSidebar({
  collapsed,
  onToggleCollapsed,
  onNewChat,
  onSelect,
  onOpenProject,
  onOpenUpgrade,
  onOpenAuth,
  user,
  profile,
  projects,
  cloudLoading,
}) {
  const recentProjects = projects.slice(0, 5);
  const pinnedProjects = projects.slice(0, 2);
  const userName = displayNameFromProfile(profile, user);
  const userPlan = user ? `${profile?.plan ?? "free"} plan` : "Sign in to sync";
  const avatarSrc = avatarFromProfile(profile, user);
  const [upgradeCardVisible, setUpgradeCardVisible] = useState(() => {
    try {
      return window.localStorage.getItem("mindi_home_upgrade_dismissed") !== "true";
    } catch {
      return true;
    }
  });

  function dismissUpgradeCard() {
    try {
      window.localStorage.setItem("mindi_home_upgrade_dismissed", "true");
    } catch {
      // The card can still close for the current session.
    }
    setUpgradeCardVisible(false);
  }

  return (
    <motion.aside
      className={collapsed ? "home-sidebar collapsed" : "home-sidebar"}
      aria-label="Home side menu"
      initial={{ x: -18, opacity: 0 }}
      animate={{ x: 0, opacity: 1, width: collapsed ? 72 : 248 }}
      transition={{ duration: 0.32, ease: "easeOut" }}
    >
      <header className="home-sidebar-head">
        <img className="home-sidebar-logo" src="/assets/mindigenous-text-logo.png" alt="MINDIGENOUS" />
        <button
          type="button"
          className="home-sidebar-collapse"
          onClick={onToggleCollapsed}
          aria-label={collapsed ? "Expand side menu" : "Collapse side menu"}
          aria-expanded={!collapsed}
        >
          {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
        </button>
      </header>

      <button type="button" className="home-sidebar-new" onClick={onNewChat} aria-label="New Chat">
        <MessageSquare size={16} />
        <span className="home-sidebar-label">New Chat</span>
      </button>

      <nav className="home-sidebar-nav" aria-label="Home navigation">
        <button type="button" className="active" aria-label="Home">
          <Home size={17} />
          <span className="home-sidebar-label">Home</span>
        </button>
        <button type="button" aria-label="Explore MINDIGENOUS AI" onClick={() => onSelect("Explore MINDIGENOUS AI features for web building.")}>
          <Search size={17} />
          <span className="home-sidebar-label">Explore MINDIGENOUS AI</span>
        </button>
      </nav>

      <section className="home-sidebar-section" aria-label="Folders">
        <header>
          <span className="home-sidebar-label">Projects</span>
          <Plus size={16} />
        </header>
        {pinnedProjects.map((project) => (
          <button type="button" key={project.id} aria-label={project.name} onClick={() => onOpenProject(project)}>
            <Folder size={17} />
            <span className="home-sidebar-label">{project.name}</span>
          </button>
        ))}
        {!pinnedProjects.length && (
          <p className="home-sidebar-empty">{cloudLoading ? "Loading projects" : user ? "No cloud projects yet" : "Sign in to load projects"}</p>
        )}
      </section>

      <section className="home-sidebar-section home-sidebar-history" aria-label="History">
        <header>
          <span className="home-sidebar-label">Recent workspaces</span>
        </header>
        {recentProjects.map((project) => (
          <button type="button" key={project.id} aria-label={project.name} onClick={() => onOpenProject(project)}>
            <MessageSquare size={17} />
            <span className="home-sidebar-label">{project.name}</span>
          </button>
        ))}
        {!recentProjects.length && (
          <p className="home-sidebar-empty">{cloudLoading ? "Syncing workspaces" : user ? "No recent workspaces" : "Sign in to restore workspaces"}</p>
        )}
      </section>

      {upgradeCardVisible && (
        <div className="home-upgrade-card">
          <button type="button" aria-label="Dismiss plan card" onClick={dismissUpgradeCard}>
            <X size={14} />
          </button>
          <strong>MINDIGENOUS</strong>
          <p>Priority processing, custom AI models, and expanded cloud storage.</p>
          <button type="button" className="home-upgrade-action" onClick={onOpenUpgrade}>Upgrade Plan</button>
        </div>
      )}

      <button type="button" className="home-user-row" aria-label={`${userName} ${userPlan}`} onClick={user ? undefined : onOpenAuth}>
        <img src={avatarSrc} alt="" />
        <span>
          <strong>{userName}</strong>
          <small>{userPlan}</small>
        </span>
        <ChevronDown size={15} />
      </button>
    </motion.aside>
  );
}

function HomeBelowPrompt({ onSelect, projects, cloudLoading, onOpenProject, promptOptionsOpen = false }) {
  const weather = useLocalWeather();
  const [recentOpen, setRecentOpen] = useState(true);
  const [weatherOpen, setWeatherOpen] = useState(false);
  const actions = [
    {
      icon: <ImagePlus size={15} />,
      label: "Create image",
      prompt: "Create a polished product hero image for an AI website builder.",
    },
    {
      icon: <FileText size={15} />,
      label: "Make a plan",
      prompt: "Make a deployment-ready plan for a modern SaaS dashboard.",
    },
    {
      icon: <Sparkles size={15} />,
      label: "Summarize text",
      prompt: "Summarize this product idea and turn it into a clear website brief.",
    },
    {
      icon: <PenLine size={15} />,
      label: "Help me write",
      prompt: "Write concise homepage copy for a premium AI developer tool.",
    },
    {
      icon: <Lightbulb size={15} />,
      label: "Brainstorm",
      prompt: "Brainstorm five professional app UI concepts with layout notes.",
    },
  ];

  const recentProjects = projects.slice(0, 3);
  const recentExpanded = recentOpen && !promptOptionsOpen;

  useEffect(() => {
    if (promptOptionsOpen) setRecentOpen(false);
  }, [promptOptionsOpen]);

  return (
    <motion.div
      className={`home-support ${promptOptionsOpen ? "prompt-options-open" : ""}`}
      initial={{ y: 14, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.3, ease: "easeOut", delay: 0.08 }}
    >
      <div className="home-action-strip" aria-label="Prompt shortcuts">
        {actions.map((action) => (
          <button type="button" className="home-action-chip" key={action.label} onClick={() => onSelect(action.prompt)}>
            {action.icon}
            <span>{action.label}</span>
          </button>
        ))}
      </div>

      <div className="home-insights">
        <button type="button" className="home-insight-card weather" onClick={() => setWeatherOpen(true)} aria-label={`Open weather forecast for ${weather.location}`}>
          <div>
            <strong>{weather.location}</strong>
            <span>{weather.source === "loading" ? "Finding location" : "Local weather"}</span>
          </div>
          <WeatherIcon code={weather.code} className="home-weather-icon" />
          <p>{weather.temperature === null ? "--" : weather.temperature}° C</p>
          <small>
            {weather.condition}
            {weather.high !== null && weather.low !== null ? ` H:${weather.high} L:${weather.low}` : ""}
          </small>
        </button>

        <article className="home-insight-card context">
          <span className="home-pill">New</span>
          <strong>Context-aware build chat</strong>
          <p>MINDIGENOUS keeps the active files, preview, and design settings aligned while you iterate.</p>
        </article>
      </div>

      <section className="home-recent" aria-label="Recent workspaces">
        <header>
          <button
            type="button"
            className="home-recent-toggle"
            onClick={() => setRecentOpen((open) => !open)}
            aria-expanded={recentExpanded}
            aria-controls="home-recent-workspaces"
          >
            <MessageSquare size={15} />
            <span>Your recent workspaces</span>
            <ChevronRight size={14} />
          </button>
        </header>
        <AnimatePresence initial={false}>
          {recentExpanded && (
            <motion.div
              id="home-recent-workspaces"
              className="home-recent-grid"
              initial={{ opacity: 0, y: -10, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -8, scale: 0.98 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              {recentProjects.map((project) => (
                <button type="button" className="home-recent-card" key={project.id} onClick={() => onOpenProject(project)}>
                  <MessageSquare size={15} />
                  <strong>{project.name}</strong>
                  <span>{formatProjectTime(project.last_opened_at ?? project.updated_at ?? project.created_at)}</span>
                </button>
              ))}
              {!recentProjects.length && (
                <div className="home-recent-empty">
                  {cloudLoading ? "Loading cloud workspaces" : "No cloud workspaces yet"}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </section>
      <AnimatePresence>
        {weatherOpen && (
          <WeatherForecastModal weather={weather} onClose={() => setWeatherOpen(false)} />
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function WeatherForecastModal({ weather, onClose }) {
  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape") onClose();
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const today = weather.forecast?.[0];

  return (
    <motion.div
      className="weather-modal-backdrop"
      role="presentation"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <motion.section
        className="weather-modal"
        role="dialog"
        aria-modal="true"
        aria-label={`${weather.location} weather forecast`}
        initial={{ y: 18, opacity: 0, scale: 0.98 }}
        animate={{ y: 0, opacity: 1, scale: 1 }}
        exit={{ y: 12, opacity: 0, scale: 0.98 }}
        transition={{ duration: 0.22, ease: "easeOut" }}
      >
        <header className="weather-modal-head">
          <div>
            <span>Live weather</span>
            <strong>{weather.location}</strong>
          </div>
          <button type="button" onClick={onClose} aria-label="Close weather forecast">
            <X size={16} />
          </button>
        </header>

        <div className="weather-modal-current">
          <WeatherIcon code={weather.code} className="weather-modal-icon" />
          <div>
            <strong>{weather.temperature === null ? "--" : weather.temperature}° C</strong>
            <span>{weather.condition}</span>
          </div>
          <small>{today ? `H:${today.high} L:${today.low}` : "Forecast loading"}</small>
        </div>

        <div className="weather-modal-metrics">
          <span>Humidity <strong>{weather.humidity === null ? "--" : `${weather.humidity}%`}</strong></span>
          <span>Wind <strong>{weather.wind === null ? "--" : `${weather.wind} km/h`}</strong></span>
          <span>Rain <strong>{today?.precipitation ?? "--"}%</strong></span>
        </div>

        <div className="weather-forecast-list">
          {(weather.forecast ?? []).map((day) => (
            <div className="weather-forecast-row" key={day.date}>
              <span>{new Date(`${day.date}T00:00:00`).toLocaleDateString(undefined, { weekday: "short" })}</span>
              <WeatherIcon code={day.code} className="weather-row-icon" />
              <strong>{day.condition}</strong>
              <small>{day.high}° / {day.low}°</small>
            </div>
          ))}
          {!weather.forecast?.length && (
            <div className="weather-forecast-empty">Live forecast unavailable right now</div>
          )}
        </div>
      </motion.section>
    </motion.div>
  );
}

function ChatMessages({ messages }) {
  if (!messages.length) {
    return (
      <div className="empty-conversation">
        <span>No messages yet</span>
      </div>
    );
  }

  return (
    <div className="messages" aria-live="polite">
      {messages.map((message) => (
        <article className={`message ${message.role}`} key={message.id}>
          <div className="message-avatar">{message.role === "assistant" ? <Bot size={16} /> : "U"}</div>
          <p>{message.content}</p>
        </article>
      ))}
    </div>
  );
}

function MinimalChatPanel({ messages }) {
  return (
    <section className="minimal-chat-card" aria-label="Chat">
      <div className="minimal-chat-head">
        <strong>Chat</strong>
        <span>MINDIGENOUS</span>
      </div>
      <div className="chat-scroll">
        <ChatMessages messages={messages} />
      </div>
    </section>
  );
}

function WorkspacePreview({ files, refreshKey, onOpenTools }) {
  return (
    <section className="preview-workspace" aria-label="Live preview">
      <header className="preview-workspace-head">
        <div>
          <span>Preview</span>
          <strong>Live output</strong>
        </div>
        <button className="tools-open-button" type="button" onClick={onOpenTools}>
          <FileCode2 size={15} />
          Files & code
        </button>
      </header>
      <PreviewPanel files={files} visible refreshKey={refreshKey} />
    </section>
  );
}

function Sidebar({
  collapsed,
  setCollapsed,
  files,
  activeFile,
  setActiveFile,
  history,
  onHome,
  editorView,
  setEditorView,
}) {
  function openFile(file) {
    setActiveFile(file);
    setEditorView("code");
  }

  return (
    <motion.aside
      className={collapsed ? "sidebar collapsed" : "sidebar"}
      initial={{ x: -260 }}
      animate={{ x: 0 }}
      transition={layoutTransition}
    >
      <div className="sidebar-head">
        {!collapsed && <LiquidLogo compact />}
        <IconButton onClick={() => setCollapsed(!collapsed)} aria-label="Toggle sidebar">
          {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
        </IconButton>
      </div>

      {!collapsed && (
        <>
          <button className="home-row" onClick={onHome}>
            <Home size={15} />
            Home
          </button>

          <section className="sidebar-section file-explorer" aria-label="File explorer">
            <div className="section-title explorer-title">
              <Folder size={14} />
              Explorer
            </div>

            <div className="explorer-tree">
              <div className="explorer-root">
                <Folder size={14} />
                <span>MINDIGENOUS</span>
              </div>

              <div className="explorer-group">
                <div className="explorer-folder">
                  <Folder size={13} />
                  <span>generated</span>
                </div>

                {Object.keys(files).map((file) => (
                  <button
                    className={activeFile === file && editorView === "code" ? "file-row active" : "file-row"}
                    key={file}
                    onClick={() => openFile(file)}
                  >
                    <FileCode2 size={15} />
                    {file}
                  </button>
                ))}
              </div>

              <button
                className={editorView === "preview" ? "file-row preview-row active" : "file-row preview-row"}
                onClick={() => setEditorView("preview")}
              >
                <Terminal size={15} />
                Live preview
              </button>
            </div>
          </section>

          <section className="sidebar-section">
            <div className="section-title">
              <History size={14} />
              History
            </div>
            {history.map((item) => (
              <button className="history-row" key={item}>{item}</button>
            ))}
          </section>
        </>
      )}
    </motion.aside>
  );
}

function ToolsWindow({
  collapsed,
  setCollapsed,
  files,
  setFiles,
  activeFile,
  setActiveFile,
  history,
  onHome,
  editorView,
  setEditorView,
  refreshKey,
  onClose,
}) {
  return (
    <motion.div
      className="tools-window"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2, ease: "easeOut" }}
    >
      <motion.div
        className="tools-sidebar-wrap"
        initial={{ x: -280, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: -280, opacity: 0 }}
        transition={{ duration: 0.42, ease: [0.22, 1, 0.36, 1] }}
      >
        <Sidebar
          collapsed={collapsed}
          setCollapsed={setCollapsed}
          files={files}
          activeFile={activeFile}
          setActiveFile={setActiveFile}
          history={history}
          onHome={onHome}
          editorView={editorView}
          setEditorView={setEditorView}
        />
      </motion.div>

      <motion.div
        className="tools-editor-wrap"
        initial={{ x: 360, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: 360, opacity: 0 }}
        transition={{ duration: 0.46, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="tools-window-bar">
          <span>{editorView === "preview" ? "Preview" : activeFile}</span>
          <button type="button" onClick={onClose}>Close</button>
        </div>
        <div className="right-stack tools-code-stack">
          <IdePanel
            files={files}
            setFiles={setFiles}
            activeFile={activeFile}
            setActiveFile={setActiveFile}
            editorView={editorView}
            setEditorView={setEditorView}
            refreshKey={refreshKey}
          />
        </div>
      </motion.div>
    </motion.div>
  );
}

function IdePanel({ files, setFiles, activeFile, setActiveFile, editorView, setEditorView, refreshKey }) {
  function openFile(file) {
    setActiveFile(file);
    setEditorView("code");
  }

  return (
    <motion.section
      className="ide-panel"
      aria-label="IDE panel"
      initial={{ x: 300 }}
      animate={{ x: 0 }}
      transition={layoutTransition}
    >
      <div className="tabs">
        {fileOrder.map((file) => (
          <button
            className={activeFile === file && editorView === "code" ? "tab active" : "tab"}
            key={file}
            onClick={() => openFile(file)}
          >
            {file}
          </button>
        ))}
        <button
          className={editorView === "preview" ? "tab preview-tab active" : "tab preview-tab"}
          onClick={() => setEditorView("preview")}
        >
          Preview
        </button>
      </div>

      <div className="ide-tools">
        <span>
          <Code2 size={14} />
          {editorView === "preview" ? "Live preview" : activeFile}
        </span>
      </div>

      {editorView === "preview" ? (
        <PreviewPanel files={files} visible refreshKey={refreshKey} />
      ) : (
        <CodeEditorWithLines
          file={activeFile}
          value={files[activeFile]}
          onChange={(nextValue) => setFiles((current) => ({ ...current, [activeFile]: nextValue }))}
        />
      )}
    </motion.section>
  );
}

function CodeEditorWithLines({ file, value = "", onChange, zoom = 88 }) {
  const lineNumberRef = useRef(null);
  const highlightRef = useRef(null);
  const lines = Math.max(1, value.split("\n").length);
  const editorScale = Math.max(0.78, Math.min(1.24, zoom / 88));
  const highlighted = useMemo(() => highlightCode(value, file), [file, value]);

  function handleScroll(event) {
    if (lineNumberRef.current) {
      lineNumberRef.current.scrollTop = event.currentTarget.scrollTop;
    }
    if (highlightRef.current) {
      highlightRef.current.scrollTop = event.currentTarget.scrollTop;
      highlightRef.current.scrollLeft = event.currentTarget.scrollLeft;
    }
  }

  return (
    <div className="code-editor-frame" data-file={file} style={{ "--editor-scale": editorScale }}>
      <div className="code-line-numbers" ref={lineNumberRef} aria-hidden="true">
        {Array.from({ length: lines }, (_, index) => (
          <span key={`${file}-line-${index + 1}`}>{index + 1}</span>
        ))}
      </div>
      <div className="editor-code-layer">
        <pre
          className="code-highlight"
          ref={highlightRef}
          aria-hidden="true"
          dangerouslySetInnerHTML={{ __html: highlighted }}
        />
        <textarea
          className="code-editor"
          value={value}
          spellCheck={false}
          onScroll={handleScroll}
          onChange={(event) => onChange(event.target.value)}
          aria-label={`${file} editor`}
        />
      </div>
    </div>
  );
}

function PreviewPanel({ files, visible, refreshKey }) {
  if (!visible) {
    return (
      <section className="preview-panel empty">
        <Terminal size={22} />
      </section>
    );
  }

  return (
    <section className="preview-panel" aria-label="Live preview">
      <iframe
        key={refreshKey}
        title="Generated preview"
        sandbox="allow-scripts"
        srcDoc={composePreview(files)}
      />
    </section>
  );
}

function WorkspaceTopChrome({
  editorView,
  setEditorView,
  onHome,
  projectName,
  projectMenuOpen,
  setProjectMenuOpen,
  onProjectSelect,
  onProjectRename,
  projects = [],
  onInvite,
  onManageAccess,
  onNotify,
}) {
  const { openAuth } = useAuthModal();
  const { user, signOut } = useAuthSession();
  const [draftName, setDraftName] = useState(projectName);
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const [inviteMenuOpen, setInviteMenuOpen] = useState(false);
  const topbarMenuRef = useRef(null);
  const userName = user?.user_metadata?.full_name || user?.email?.split("@")[0] || "Account";
  const userEmail = user?.email ?? "Signed in";
  const avatarSrc = user?.user_metadata?.avatar_url || userAvatar;

  useEffect(() => {
    setDraftName(projectName);
  }, [projectName]);

  useEffect(() => {
    function handlePointerDown(event) {
      if (!topbarMenuRef.current?.contains(event.target)) {
        setAccountMenuOpen(false);
        setInviteMenuOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, []);

  function handleRename(event) {
    event.preventDefault();
    const nextName = draftName.trim();
    if (!nextName) return;
    onProjectRename(nextName);
    setProjectMenuOpen(false);
  }

  async function handleSignOut() {
    const result = await signOut();
    setAccountMenuOpen(false);
    onNotify?.(result.message);
  }

  return (
    <motion.header
      className="workspace-global-topbar"
      initial={{ y: -36, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.42, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="workspace-global-brand">
        <button className="brand-word brand-logo-button" type="button" onClick={onHome} aria-label="MINDIGENOUS home">
          <span className="logo-shine-mask brand-logo-mask">
            <img className="brand-logo-image" src="/assets/mindigenous-text-logo.png" alt="MINDIGENOUS" />
          </span>
        </button>
        <div className="project-menu-wrap">
          <button
            className="project-select"
            type="button"
            onClick={() => setProjectMenuOpen((open) => !open)}
            aria-expanded={projectMenuOpen}
          >
            <span>{projectName}</span>
            <ChevronDown size={14} />
          </button>
          <AnimatePresence>
            {projectMenuOpen && (
              <motion.div
                className="project-menu"
                initial={{ y: -6, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ y: -6, opacity: 0 }}
                transition={{ duration: 0.16, ease: "easeOut" }}
              >
                <form className="project-rename" onSubmit={handleRename}>
                  <label htmlFor="project-rename-input">Rename project</label>
                  <div>
                    <input
                      id="project-rename-input"
                      value={draftName}
                      onChange={(event) => setDraftName(event.target.value)}
                    />
                    <button type="submit">Save</button>
                  </div>
                </form>
                {projects.map((project) => (
                  <button type="button" key={project.id} onClick={() => onProjectSelect(project)}>
                    {project.name}
                  </button>
                ))}
                {!projects.length && <span className="project-menu-empty">No cloud projects yet</span>}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <div className="mode-switch" role="tablist" aria-label="Workspace mode">
        {["Preview", "Design", "Code"].map((item) => {
          const view = item.toLowerCase();
          const active = editorView === view || (view === "code" && editorView === "terminal");
          return (
            <button
              type="button"
              key={item}
              className={active ? "active" : ""}
              onClick={() => setEditorView(view)}
            >
              {item}
            </button>
          );
        })}
      </div>

      <div className="topbar-account" ref={topbarMenuRef}>
        {user ? (
          <div className={accountMenuOpen ? "account-hover open" : "account-hover"}>
            <button
              className="avatar-button"
              type="button"
              onClick={() => {
                setAccountMenuOpen((open) => !open);
                setInviteMenuOpen(false);
              }}
              aria-expanded={accountMenuOpen}
              aria-label="Account menu"
            >
              <img className="avatar-dot" src={avatarSrc} alt={userName} />
            </button>
            <div className="account-popover">
              <strong>{userName}</strong>
              <span>{userEmail}</span>
              <button type="button" onClick={() => setAccountMenuOpen(false)}>Profile settings</button>
              <button type="button" onClick={() => setAccountMenuOpen(false)}>Account usage</button>
              <button type="button" onClick={handleSignOut}>Sign out</button>
            </div>
          </div>
        ) : (
          <button className="login-button" type="button" onClick={() => openAuth("signin")}>
            Login
          </button>
        )}
        <div className={inviteMenuOpen ? "invite-hover open" : "invite-hover"}>
          <button
            className="invite-button"
            type="button"
            onClick={() => {
              setInviteMenuOpen((open) => !open);
              setAccountMenuOpen(false);
            }}
            aria-expanded={inviteMenuOpen}
          >
            Invite
          </button>
          <div className="invite-popover">
            <button
              type="button"
              onClick={() => {
                onInvite();
                setInviteMenuOpen(false);
              }}
            >
              Copy invite link
            </button>
            <button
              type="button"
              onClick={() => {
                onManageAccess?.();
                setInviteMenuOpen(false);
              }}
            >
              Manage access
            </button>
          </div>
        </div>
      </div>
    </motion.header>
  );
}

function WorkspaceRail({ activeTool, onToolChange, onHome, onRun, onExport, onDeploy }) {
  const items = [
    { id: "explorer", label: "Explorer", icon: <Folder size={18} /> },
    { id: "extensions", label: "Extensions", icon: <Blocks size={18} /> },
    { id: "settings", label: "Settings", icon: <Settings size={18} /> },
    { id: "repository", label: "Repository", icon: <GitBranch size={18} /> },
  ];

  return (
    <motion.nav
      className="workspace-rail"
      aria-label="Workspace rail"
      initial={{ x: -48, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.42, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="rail-stack">
        {items.map((item) => (
          <button
            className={activeTool === item.id ? "rail-button active" : "rail-button"}
            type="button"
            key={item.id}
            onClick={() => onToolChange(item.id)}
            aria-label={item.label}
          >
            {item.icon}
          </button>
        ))}
      </div>
      <div className="rail-stack rail-bottom-stack">
        <button className="rail-button" type="button" onClick={onRun} aria-label="Run">
          <Play size={17} />
        </button>
        <button className="rail-button" type="button" onClick={onExport} aria-label="Export">
          <Download size={17} />
        </button>
        <button className="rail-button crown" type="button" onClick={onDeploy} aria-label="Deploy">
          <Rocket size={17} />
        </button>
        <button className="rail-button" type="button" onClick={onHome} aria-label="Home">
          <Home size={17} />
        </button>
      </div>
    </motion.nav>
  );
}

function ExplorerPanel({
  files,
  activeFile,
  editorView,
  history,
  activeTool,
  onOpenFile,
  onOpenTerminal,
  terminalOpen,
  onCreateFile,
  onCreateFolder,
  onOpenSearch,
  onNotify,
  customFolders = [],
}) {
  const [createMenuOpen, setCreateMenuOpen] = useState(false);
  const [openFolders, setOpenFolders] = useState({
    generated: true,
    src: true,
    styles: true,
    scripts: true,
    components: false,
  });

  function toggleFolder(id) {
    setOpenFolders((current) => ({ ...current, [id]: !current[id] }));
  }

  const compileFiles = fileOrder.filter((file) => files[file]);
  const projectFiles = [
    { type: "folder", id: "generated", label: "generated", depth: 0, open: openFolders.generated },
    ...(openFolders.generated ? compileFiles.map((file) => ({ type: "file", label: file, depth: 1, generated: true })) : []),
  ];

  return (
    <motion.aside
      className="workspace-explorer"
      aria-label="File explorer"
      initial={{ x: -80, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.48, delay: 0.08, ease: [0.22, 1, 0.36, 1] }}
    >
      <section className="explorer-section">
        <div className="explorer-heading">
          <strong>{activeTool === "explorer" ? "Explorer" : activeTool === "repository" ? "Repository" : activeTool === "settings" ? "Settings" : "Extensions"}</strong>
          <div className="explorer-heading-actions">
            <button type="button" onClick={() => setCreateMenuOpen((open) => !open)} aria-label="New file or folder">
              <Plus size={14} />
            </button>
            <button type="button" onClick={onOpenSearch} aria-label="Search files">
              <Search size={15} />
            </button>
            <AnimatePresence>
              {createMenuOpen && (
                <motion.div
                  className="create-menu"
                  initial={{ y: -5, opacity: 0, scale: 0.98 }}
                  animate={{ y: 0, opacity: 1, scale: 1 }}
                  exit={{ y: -5, opacity: 0, scale: 0.98 }}
                  transition={{ duration: 0.16, ease: "easeOut" }}
                >
                  <button
                    type="button"
                    onClick={() => {
                      setCreateMenuOpen(false);
                      onCreateFile();
                    }}
                  >
                    New file
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setCreateMenuOpen(false);
                      onCreateFolder();
                    }}
                  >
                    New folder
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
        <div className="workspace-tree">
          {activeTool === "extensions" ? (
            <ExtensionsPanel onNotify={onNotify} />
          ) : activeTool === "settings" ? (
            <SettingsPanel onNotify={onNotify} />
          ) : activeTool === "repository" ? (
            <RepositoryPanel onOpenFile={onOpenFile} />
          ) : (
            <>
              {projectFiles.map((item, index) => item.type === "folder" ? (
                <TreeFolder
                  key={`${item.label}-${index}`}
                  label={item.label}
                  depth={item.depth}
                  open={item.open}
                  onClick={() => toggleFolder(item.id)}
                />
              ) : (
                <TreeFile
                  key={`${item.label}-${index}`}
                  label={item.label}
                  depth={item.depth}
                  kind={fileKind(item.label)}
                  active={activeFile === (item.target ?? item.label) && editorView === "code"}
                  onClick={() => onOpenFile(item.target ?? item.label)}
                />
              ))}
              <button
                className={terminalOpen ? "tree-file preview terminal-entry active" : "tree-file preview terminal-entry"}
                type="button"
                onClick={onOpenTerminal}
              >
                <Terminal size={14} />
                <span>Terminal</span>
              </button>
            </>
          )}
        </div>
      </section>
    </motion.aside>
  );
}

function TreeFolder({ label, depth = 0, open = false, onClick }) {
  return (
    <button className="tree-folder" style={{ "--depth": depth }} type="button" onClick={onClick}>
      <ChevronRight size={13} className={open ? "open" : ""} />
      <Folder size={14} />
      <span>{label}</span>
    </button>
  );
}

function TreeFile({ label, depth = 0, active = false, onClick, kind = "tsx" }) {
  return (
    <button
      className={active ? `tree-file ${kind} active` : `tree-file ${kind}`}
      style={{ "--depth": depth }}
      type="button"
      onClick={onClick}
    >
      <FileIcon file={label} kind={kind} />
      <span>{label}</span>
    </button>
  );
}

function ExtensionsPanel({ onNotify }) {
  const [enabled, setEnabled] = useState(["React Tools", "Prettier", "Tailwind CSS"]);
  const extensions = [
    { name: "React Tools", meta: "Components, hooks, JSX hints" },
    { name: "Tailwind CSS", meta: "Utility autocomplete and color preview" },
    { name: "Prettier", meta: "Format generated code on save" },
    { name: "ESLint", meta: "Catch broken imports and unused logic" },
    { name: "Vite Runner", meta: "Fast preview, build, and refresh tasks" },
  ];

  function toggleExtension(name) {
    setEnabled((current) => {
      const active = current.includes(name);
      onNotify?.(`${name} ${active ? "disabled" : "enabled"}`);
      return active ? current.filter((item) => item !== name) : [...current, name];
    });
  }

  return (
    <div className="extension-list">
      {extensions.map((extension) => {
        const active = enabled.includes(extension.name);
        return (
          <button
            type="button"
            key={extension.name}
            className={active ? "extension-card active" : "extension-card"}
            onClick={() => toggleExtension(extension.name)}
          >
            <Sparkles size={14} />
            <span>
              <strong>{extension.name}</strong>
              <small>{extension.meta}</small>
            </span>
            <em>{active ? "On" : "Off"}</em>
          </button>
        );
      })}
    </div>
  );
}

function SettingsPanel({ onNotify }) {
  const [settings, setSettings] = useState({
    autosave: true,
    minimap: false,
    formatOnSave: true,
    compactMode: false,
  });

  function toggleSetting(key) {
    setSettings((current) => {
      const next = !current[key];
      onNotify?.(`${key.replace(/([A-Z])/g, " $1")} ${next ? "enabled" : "disabled"}`);
      return { ...current, [key]: next };
    });
  }

  const rows = [
    ["autosave", "Auto save", "Persist generated code changes"],
    ["formatOnSave", "Format on save", "Clean HTML/CSS/JS before export"],
    ["minimap", "Editor minimap", "Show code structure guide"],
    ["compactMode", "Compact panels", "Reduce sidebar and assistant density"],
  ];

  return (
    <div className="settings-panel">
      {rows.map(([key, label, meta]) => (
        <button
          type="button"
          key={key}
          className={settings[key] ? "settings-row active" : "settings-row"}
          onClick={() => toggleSetting(key)}
        >
          <span>
            <strong>{label}</strong>
            <small>{meta}</small>
          </span>
          <em>{settings[key] ? "On" : "Off"}</em>
        </button>
      ))}
    </div>
  );
}

function AppearanceSettings({ onNotify, onSaved, user }) {
  const [settings, setSettings] = useState(() => readStoredAppearanceSettings(user));
  const imageInputRef = useRef(null);
  const themes = [
    { id: "system", label: "System preference", icon: <Monitor size={14} /> },
    { id: "light", label: "Light mode", icon: <Sun size={14} /> },
    { id: "dark", label: "Dark", icon: <Moon size={14} /> },
  ];
  const overlayColors = ["#22c55e", "#8b5cf6", "#38bdf8", "#f59e0b", "#ef4444", "#a855f7", "#14b8a6", "#64748b"];

  function updateSetting(key, value) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  useEffect(() => {
    applyAppearanceSettings(settings);
  }, [settings]);

  async function saveSettings() {
    try {
      writeStoredAppearanceSettings(settings, user);
    } catch {
      // Appearance still applies in this session even when storage is unavailable.
    }
    applyAppearanceSettings(settings);
    if (user?.id) {
      const savedToCloud = await saveProfileAppearanceSettings(user.id, normalizeAppearanceSettings(settings));
      onNotify?.(savedToCloud ? "Appearance saved" : "Appearance saved locally");
    } else {
      onNotify?.("Appearance saved locally");
    }
    onSaved?.();
  }

  function resetSettings() {
    const stored = readStoredAppearanceSettings(user);
    setSettings(stored);
    applyAppearanceSettings(stored);
  }

  function handleImageUpload(event) {
    const input = event.currentTarget;
    const file = input.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      onNotify?.("Choose an image file");
      input.value = "";
      return;
    }

    const reader = new FileReader();
    reader.addEventListener("load", () => {
      const url = String(reader.result || "");
      setSettings((current) => ({
        ...current,
        backgroundEnabled: true,
        backgroundImage: "custom",
        customBackgroundUrl: url,
      }));
      onNotify?.("Background image added");
      input.value = "";
    });
    reader.readAsDataURL(file);
  }

  const selectedBackground =
    settings.backgroundImage === "custom" && settings.customBackgroundUrl
      ? { id: "custom", label: "Uploaded image", url: settings.customBackgroundUrl }
      : APPEARANCE_BACKGROUND_BY_ID[settings.backgroundImage] ?? APPEARANCE_BACKGROUND_BY_ID[DEFAULT_APPEARANCE_SETTINGS.backgroundImage];

  return (
    <section className="appearance-settings" aria-label="Appearance settings">
      <header className="appearance-head">
        <div>
          <strong>Interface theme</strong>
          <small>Select or customize your UI theme</small>
        </div>
      </header>

      <div className="appearance-theme-grid">
        {themes.map((theme) => (
          <button
            type="button"
            key={theme.id}
            className={settings.theme === theme.id ? "theme-card active" : "theme-card"}
            onClick={() => updateSetting("theme", theme.id)}
          >
            <span className={`theme-preview ${theme.id}`}>
              <i />
              <b />
              <em />
            </span>
            <span>{theme.icon}{theme.label}</span>
          </button>
        ))}
      </div>

      <div className="appearance-switch-row">
        <span>
          <strong>Background</strong>
          <small>Customize your background</small>
        </span>
        <button
          type="button"
          className={settings.backgroundEnabled ? "switch active" : "switch"}
          onClick={() => updateSetting("backgroundEnabled", !settings.backgroundEnabled)}
          aria-pressed={settings.backgroundEnabled}
        />
      </div>

      {settings.backgroundEnabled && (
        <div className="appearance-background-grid">
          <div className="appearance-preview-box">
            <div className="preview-surface" style={{ "--preview-image": `url("${selectedBackground.url}")` }} />
            <input
              ref={imageInputRef}
              className="appearance-file-input"
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              aria-label="Upload background image"
            />
            <button type="button" onClick={() => imageInputRef.current?.click()}>
              <ImagePlus size={14} /> Add image
            </button>
          </div>
          <div className="appearance-control-stack">
            <div className="appearance-control-row">
              <span>Auto generate background</span>
              <button type="button" onClick={() => onNotify?.("Background generation queued")}>
                <RefreshCw size={13} /> Generate
              </button>
            </div>
            <div className="appearance-control-row swatches">
              <span>Overlay</span>
              <div className="overlay-swatch-list">
                {overlayColors.map((color) => (
                  <button
                    type="button"
                    key={color}
                    className={settings.overlay === color ? "overlay-swatch active" : "overlay-swatch"}
                    style={{ "--swatch": color }}
                    onClick={() => updateSetting("overlay", color)}
                    aria-label={`Use ${color}`}
                  />
                ))}
                <button type="button" className="overlay-add" aria-label="Add overlay color">+</button>
              </div>
            </div>
            <div className="appearance-control-row">
              <span>Blur</span>
              <button
                type="button"
                className={settings.blur ? "switch compact active" : "switch compact"}
                onClick={() => updateSetting("blur", !settings.blur)}
                aria-pressed={settings.blur}
              />
            </div>
            <div className="appearance-control-row">
              <span>Apply to all</span>
              <button
                type="button"
                className={settings.applyAll ? "switch compact active" : "switch compact"}
                onClick={() => updateSetting("applyAll", !settings.applyAll)}
                aria-pressed={settings.applyAll}
              />
            </div>
          </div>
        </div>
      )}

      <div className="appearance-switch-row">
        <span>
          <strong>Chat color</strong>
          <small>Customize the assistant prompt surface</small>
        </span>
        <button
          type="button"
          className={settings.chatColorEnabled ? "switch active" : "switch"}
          onClick={() => updateSetting("chatColorEnabled", !settings.chatColorEnabled)}
          aria-pressed={settings.chatColorEnabled}
        />
      </div>

      <div className="landscape-row">
        <span>Landscape <button type="button">View All <ChevronRight size={12} /></button></span>
        <div>
          {APPEARANCE_BACKGROUNDS.slice(0, 4).map((background) => (
            <button
              type="button"
              key={background.id}
              className={settings.backgroundImage === background.id ? "landscape-tile active" : "landscape-tile"}
              style={{ "--tile-image": `url("${background.url}")` }}
              onClick={() => updateSetting("backgroundImage", background.id)}
              aria-label={background.label}
            />
          ))}
        </div>
      </div>

      <div className="landscape-row abstract">
        <span>Abstract <button type="button">View All <ChevronRight size={12} /></button></span>
        <div>
          {APPEARANCE_BACKGROUNDS.slice(4).map((background) => (
            <button
              type="button"
              key={background.id}
              className={settings.backgroundImage === background.id ? "landscape-tile active" : "landscape-tile"}
              style={{ "--tile-image": `url("${background.url}")` }}
              onClick={() => updateSetting("backgroundImage", background.id)}
              aria-label={background.label}
            />
          ))}
        </div>
      </div>

      <footer className="appearance-footer">
        <button type="button" onClick={resetSettings}>Cancel</button>
        <button type="button" onClick={saveSettings}>Save changes</button>
      </footer>
    </section>
  );
}

function SettingsModal({ onClose, onNotify, user }) {
  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape") onClose();
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <motion.div
      className="settings-modal-backdrop"
      onClick={onClose}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.18, ease: "easeOut" }}
    >
      <motion.section
        className="settings-modal"
        aria-label="Appearance settings"
        onClick={(event) => event.stopPropagation()}
        initial={{ y: 18, scale: 0.98, opacity: 0 }}
        animate={{ y: 0, scale: 1, opacity: 1 }}
        exit={{ y: 12, scale: 0.985, opacity: 0 }}
        transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
      >
        <header className="settings-modal-head">
          <div>
            <span><Palette size={15} /> Appearance</span>
          </div>
          <button type="button" onClick={onClose} aria-label="Close settings">
            <X size={15} />
          </button>
        </header>
        <div className="settings-modal-body enhanced">
          <AppearanceSettings onNotify={onNotify} onSaved={onClose} user={user} />
        </div>
      </motion.section>
    </motion.div>
  );
}

function HomeSettingsModal({ onClose, onNotify, onOpenAuth, onOpenUpgrade, user, profile }) {
  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape") onClose();
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const accountName = displayNameFromProfile(profile, user);
  const accountEmail = user?.email ?? "Sign in to sync projects and settings";
  const planLabel = user ? `${profile?.plan ?? "free"} plan` : "Not synced";
  const settingsGroups = [
    {
      icon: <Users size={16} />,
      title: "Account profile",
      detail: user ? "Manage identity, avatar, and active workspace owner." : "Connect an account before cloud settings can sync.",
      action: user ? "Manage" : "Sign in",
      onClick: () => {
        if (!user) onOpenAuth?.();
        else onNotify?.("Account profile settings ready");
      },
    },
    {
      icon: <ShieldCheck size={16} />,
      title: "Security",
      detail: "Password, sessions, trusted devices, and privacy controls.",
      action: "Open",
      onClick: () => onNotify?.("Security settings ready"),
    },
    {
      icon: <Rocket size={16} />,
      title: "Plan and billing",
      detail: "Subscription, invoices, usage limits, and cloud storage.",
      action: "Upgrade",
      onClick: () => {
        onClose();
        onOpenUpgrade?.();
      },
    },
    {
      icon: <Download size={16} />,
      title: "Data export",
      detail: "Download workspace metadata and generated project files.",
      action: "Export",
      onClick: () => onNotify?.("Data export will be available from cloud workspaces"),
    },
  ];

  return (
    <motion.div
      className="settings-modal-backdrop"
      onClick={onClose}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.18, ease: "easeOut" }}
    >
      <motion.section
        className="settings-modal home-settings-modal"
        aria-label="Account settings"
        onClick={(event) => event.stopPropagation()}
        initial={{ y: 18, scale: 0.98, opacity: 0 }}
        animate={{ y: 0, scale: 1, opacity: 1 }}
        exit={{ y: 12, scale: 0.985, opacity: 0 }}
        transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
      >
        <header className="settings-modal-head">
          <div>
            <span><Settings size={15} /> Settings</span>
          </div>
          <button type="button" onClick={onClose} aria-label="Close settings">
            <X size={15} />
          </button>
        </header>
        <div className="settings-modal-body enhanced home-settings-body">
          <section className="home-settings-profile">
            <img src={avatarFromProfile(profile, user)} alt="" />
            <span>
              <strong>{accountName}</strong>
              <small>{accountEmail}</small>
            </span>
            <em>{planLabel}</em>
          </section>
          <section className="home-settings-grid" aria-label="Settings options">
            {settingsGroups.map((item) => (
              <button type="button" key={item.title} onClick={item.onClick}>
                <span>{item.icon}</span>
                <span>
                  <strong>{item.title}</strong>
                  <small>{item.detail}</small>
                </span>
                <em>{item.action}</em>
              </button>
            ))}
          </section>
        </div>
      </motion.section>
    </motion.div>
  );
}

function ManageAccessModal({ onClose, onCopyInvite, onNotify }) {
  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape") onClose();
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const collaborators = [
    { name: "Faaz", email: "owner@mindigenous.local", role: "Owner" },
    { name: "Design agent", email: "agent@mindigenous.local", role: "Can edit" },
    { name: "Preview reviewer", email: "review@mindigenous.local", role: "Can view" },
  ];

  return (
    <motion.div
      className="access-modal-backdrop"
      onClick={onClose}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.18, ease: "easeOut" }}
    >
      <motion.section
        className="access-modal"
        aria-label="Manage access"
        onClick={(event) => event.stopPropagation()}
        initial={{ y: 18, scale: 0.97, opacity: 0 }}
        animate={{ y: 0, scale: 1, opacity: 1 }}
        exit={{ y: 10, scale: 0.98, opacity: 0 }}
        transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
      >
        <header className="access-modal-head">
          <div>
            <span>Project sharing</span>
            <strong>Manage access</strong>
          </div>
          <button type="button" onClick={onClose} aria-label="Close manage access">
            <X size={15} />
          </button>
        </header>

        <div className="access-link-card">
          <span><Link size={15} /> Invite link</span>
          <p>Anyone with this link can request access to this workspace.</p>
          <button
            type="button"
            onClick={() => {
              onCopyInvite?.();
              onNotify?.("Invite link copied");
            }}
          >
            <Copy size={14} /> Copy link
          </button>
        </div>

        <div className="access-people">
          <div className="access-section-title">
            <Users size={15} />
            <strong>People with access</strong>
          </div>
          {collaborators.map((person) => (
            <article className="access-person" key={person.email}>
              <span>{person.name.slice(0, 1)}</span>
              <div>
                <strong>{person.name}</strong>
                <small>{person.email}</small>
              </div>
              <button type="button">{person.role}</button>
            </article>
          ))}
        </div>

        <div className="access-policy">
          <ShieldCheck size={16} />
          <div>
            <strong>Workspace permissions</strong>
            <p>Editors can update generated files, design settings, and preview state. Viewers can inspect output only.</p>
          </div>
        </div>

        <footer className="access-modal-actions">
          <button type="button" onClick={() => onNotify?.("Invite email composer opened")}>
            <MailPlus size={14} /> Invite by email
          </button>
          <button type="button" onClick={onClose}>Done</button>
        </footer>
      </motion.section>
    </motion.div>
  );
}

function RepositoryPanel({ onOpenFile }) {
  const changes = [
    {
      file: "index.html",
      added: 18,
      removed: 3,
      lines: [
        ["+", "+ <section class=\"hero\">"],
        ["+", "+   <button id=\"primaryAction\">Preview flow</button>"],
        ["-", "- <div class=\"placeholder\"></div>"],
      ],
    },
    {
      file: "styles.css",
      added: 42,
      removed: 7,
      lines: [
        ["+", "+ .hero { min-height: 100vh; }"],
        ["+", "+ .screen-line { border-radius: 999px; }"],
        ["-", "- background: #111;"],
      ],
    },
    {
      file: "script.js",
      added: 9,
      removed: 1,
      lines: [
        ["+", "+ primaryAction.addEventListener(\"click\", handlePreview);"],
        ["-", "- console.log(\"demo\");"],
      ],
    },
  ];

  return (
    <div className="repository-panel">
      <div className="repo-summary">
        <strong>Working tree</strong>
        <span><em>+69</em> <small>-11</small></span>
      </div>
      {changes.map((change) => (
        <article className="repo-change" key={change.file}>
          <button type="button" onClick={() => onOpenFile(change.file)}>
            <FileIcon file={change.file} />
            <span>{change.file}</span>
            <small>+{change.added} -{change.removed}</small>
          </button>
          <pre>
            {change.lines.map(([kind, text]) => (
              <code className={kind === "+" ? "diff-add" : "diff-remove"} key={text}>{text}</code>
            ))}
          </pre>
        </article>
      ))}
    </div>
  );
}

function LegacyExplorerPanel({ files, activeFile, setActiveFile, editorView, setEditorView, history }) {
  function openFile(file) {
    setActiveFile(file);
    setEditorView("code");
  }

  return (
    <motion.aside
      className="workspace-explorer"
      aria-label="File explorer"
      initial={{ x: -80, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.48, delay: 0.08, ease: [0.22, 1, 0.36, 1] }}
    >
      <section className="explorer-section">
        <div className="explorer-label">Explorer</div>
        <div className="workspace-tree">
          <div className="tree-folder">
            <Folder size={13} />
            <span>generated</span>
          </div>
          {Object.keys(files).map((file) => (
            <button
              className={activeFile === file && editorView === "code" ? "tree-file active" : "tree-file"}
              type="button"
              key={file}
              onClick={() => openFile(file)}
            >
              <FileCode2 size={14} />
              <span>{file}</span>
            </button>
          ))}
          <button
            className={editorView === "preview" ? "tree-file preview active" : "tree-file preview"}
            type="button"
            onClick={() => setEditorView("preview")}
          >
            <Terminal size={14} />
            <span>Live preview</span>
          </button>
        </div>
      </section>

      <section className="explorer-section history-list">
        <div className="explorer-label">Recent</div>
        {history.slice(0, 5).map((item) => (
          <button className="recent-item" type="button" key={item}>
            {item}
          </button>
        ))}
      </section>
    </motion.aside>
  );
}

function CodeWorkspace({
  files,
  setFiles,
  activeFile,
  setActiveFile,
  editorView,
  setEditorView,
  designSettings,
  onDesignSettingsChange,
  refreshKey,
  openTabs,
  onOpenFile,
  onCloseTab,
  onCreateFile,
  onCopyCode,
  zoom,
  onZoomChange,
  fullscreen,
  onToggleFullscreen,
  terminalOpen,
}) {
  const tabs = openTabs.filter((file) => fileOrder.includes(file) && files[file]);
  const isDesignView = editorView === "design";

  return (
    <motion.section
      className={[
        "code-workspace",
        fullscreen ? "fullscreen" : "",
        isDesignView ? "design-view" : "",
        editorView === "preview" ? "preview-view" : "",
      ].filter(Boolean).join(" ")}
      aria-label="Code workspace"
      layout
      initial={{ scale: 0.985, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{
        duration: 0.5,
        delay: 0.14,
        ease: [0.22, 1, 0.36, 1],
        layout: { duration: 0.36, ease: [0.22, 1, 0.36, 1] },
      }}
    >
      {editorView === "code" && (
        <div className="workspace-tabbar" role="tablist" aria-label="Editor tabs">
          {tabs.map((file) => (
            <button
              className={activeFile === file && editorView === "code" ? `workspace-tab ${fileKind(file)} active` : `workspace-tab ${fileKind(file)}`}
              type="button"
              key={file}
              onClick={() => onOpenFile(file)}
            >
              <FileIcon file={file} />
              {file}
              <span
                className="tab-close"
                role="button"
                tabIndex={0}
                aria-label={`Close ${file}`}
                onClick={(event) => {
                  event.stopPropagation();
                  onCloseTab(file);
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    event.stopPropagation();
                    onCloseTab(file);
                  }
                }}
              >
                <X size={12} />
              </span>
            </button>
          ))}
          <button className="workspace-tab icon-only" type="button" aria-label="Add file" onClick={onCreateFile}>
            <Plus size={15} />
          </button>
          <div className="editor-mini-tools">
            <button type="button" onClick={onZoomChange}>{zoom}% <ChevronDown size={12} /></button>
            <button type="button" aria-label="Copy" onClick={() => onCopyCode(files[activeFile] ?? "")}>
              <Copy size={14} />
            </button>
            <button type="button" aria-label="Maximize" onClick={onToggleFullscreen}>
              <Maximize2 size={14} />
            </button>
          </div>
        </div>
      )}

      <div className="workspace-editor-body">
        {editorView === "preview" ? (
          <PreviewPanel files={files} visible refreshKey={refreshKey} />
        ) : editorView === "design" ? (
          <DesignPanel
            files={files}
            settings={designSettings}
            onSettingsChange={onDesignSettingsChange}
          />
        ) : (
          <>
            <div className="editor-path">
              <Code2 size={14} />
              <span>generated / {activeFile}</span>
            </div>
            <CodeEditorWithLines
              file={activeFile}
              value={files[activeFile]}
              onChange={(nextValue) => setFiles((current) => ({ ...current, [activeFile]: nextValue }))}
              zoom={zoom}
            />
            <AnimatePresence>
              {terminalOpen && (
                <motion.div
                  className="terminal-drawer"
                  initial={{ y: "100%", opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  exit={{ y: "100%", opacity: 0 }}
                  transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
                >
                  <TerminalPanel activeFile={activeFile} compact />
                </motion.div>
              )}
            </AnimatePresence>
          </>
        )}
      </div>
    </motion.section>
  );
}

function DesignPanel({ files, settings, onSettingsChange }) {
  const normalizedSettings = normalizeDesignSettings(settings);

  function updateSetting(key, value) {
    const nextValue = key === "radius" || key === "spacing" ? Number(value) : value;
    onSettingsChange(normalizeDesignSettings({ ...normalizedSettings, [key]: nextValue }));
  }

  return (
    <section className="design-panel" aria-label="Design canvas">
      <div className="design-preview-pane">
        <PreviewPanel files={files} visible refreshKey="design" />
      </div>
      <aside className="design-properties">
        <strong>Design controls</strong>
        <label>
          Headline
          <input value={normalizedSettings.headline} onChange={(event) => updateSetting("headline", event.target.value)} />
        </label>
        <label>
          Body copy
          <textarea value={normalizedSettings.body} onChange={(event) => updateSetting("body", event.target.value)} />
        </label>
        <label>
          CTA text
          <input value={normalizedSettings.cta} onChange={(event) => updateSetting("cta", event.target.value)} />
        </label>
        <div className="design-two-up">
          <label>
            Accent
            <input type="color" value={normalizedSettings.accent} onChange={(event) => updateSetting("accent", event.target.value)} />
          </label>
          <label>
            Surface
            <input type="color" value={normalizedSettings.background} onChange={(event) => updateSetting("background", event.target.value)} />
          </label>
        </div>
        <label>
          Font
          <select value={normalizedSettings.font} onChange={(event) => updateSetting("font", event.target.value)}>
            <option>Inter</option>
            <option>Geist</option>
            <option>Manrope</option>
            <option>Space Grotesk</option>
          </select>
        </label>
        <label>
          Radius
          <input type="range" min="4" max="28" value={normalizedSettings.radius} onChange={(event) => updateSetting("radius", event.target.value)} />
        </label>
        <label>
          Spacing
          <input type="range" min="12" max="56" value={normalizedSettings.spacing} onChange={(event) => updateSetting("spacing", event.target.value)} />
        </label>
        <button type="button" onClick={() => onSettingsChange(normalizedSettings)}>
          Live synced to HTML/CSS
        </button>
      </aside>
    </section>
  );
}

function TerminalPanel({ activeFile, compact = false }) {
  return (
    <section className={compact ? "terminal-panel compact" : "terminal-panel"} aria-label="Workspace terminal">
      <div className="terminal-title">
        <Terminal size={14} />
        <span>Terminal</span>
      </div>
      <pre>{`mindi@workspace:~$ npm run build
vite v6.0.0 building for production...
transforming modules...
generated/${activeFile} linked
build ready in 412ms

mindi@workspace:~$ _`}</pre>
    </section>
  );
}

function FollowUpPrompt({ value, setValue, onSubmit, busy, promptFrameRef }) {
  const models = ["MINDI 1.5", "MINDI 1.0", "GPT 5.5", "Claude Opus", "DeepSeek Coder"];
  const [selectedModel, setSelectedModel] = useState("MINDI 1.5");
  const [modelMenuOpen, setModelMenuOpen] = useState(false);

  function handleSubmit(event) {
    event.preventDefault();
    onSubmit(event);
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      handleSubmit(event);
    }
  }

  return (
    <motion.form
      ref={promptFrameRef}
      className={busy ? "assistant-followup is-busy" : "assistant-followup"}
      onSubmit={handleSubmit}
      initial={{ y: 42, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.42, delay: 0.34, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="followup-model-wrap">
        <button
          className="followup-model-button"
          type="button"
          onClick={() => setModelMenuOpen((open) => !open)}
          aria-expanded={modelMenuOpen}
        >
          <Bot size={13} />
          {selectedModel}
          <ChevronDown size={12} />
        </button>
        <AnimatePresence>
          {modelMenuOpen && (
            <motion.div
              className="followup-model-menu"
              initial={{ y: 6, opacity: 0, scale: 0.98 }}
              animate={{ y: 0, opacity: 1, scale: 1 }}
              exit={{ y: 6, opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.16, ease: "easeOut" }}
            >
              {models.map((model) => (
                <button
                  type="button"
                  key={model}
                  className={model === selectedModel ? "active" : ""}
                  onClick={() => {
                    setSelectedModel(model);
                    setModelMenuOpen(false);
                  }}
                >
                  <span>{model}</span>
                  {model === selectedModel && <CheckCircle2 size={13} />}
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      <textarea
        value={value}
        rows={3}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask a follow-up..."
      />
      <div className="followup-actions">
        <div className="followup-tools">
          <button
            type="button"
            aria-label="Upload files"
            onClick={() => setValue(value ? `${value}\nUse the uploaded files as source context.` : "Use the uploaded files as source context.")}
          >
            <Paperclip size={15} />
          </button>
          <button
            type="button"
            aria-label="Enhance prompt"
            onClick={() => setValue(value ? `${value}\nMake it production-ready and responsive.` : "Make this production-ready and responsive.")}
          >
            <Sparkles size={15} />
          </button>
          <button
            type="button"
            aria-label="Voice input"
            onClick={() => setValue(value ? `${value}\nVoice input placeholder enabled.` : "Voice input placeholder enabled.")}
          >
            <Mic size={15} />
          </button>
        </div>
        <button className="followup-send" type="submit" disabled={!value.trim() || busy} aria-label="Send follow-up">
          <Rocket size={15} />
        </button>
      </div>
    </motion.form>
  );
}

function AssistancePanel({
  messages,
  activeFile,
  busy,
  agentEvents = [],
  isAgentStreaming = false,
  prompt,
  setPrompt,
  onSubmit,
  promptFrameRef,
  collapsed,
  onToggleCollapsed,
}) {
  const latestUser = [...messages].reverse().find((message) => message.role === "user");
  const latestAssistant = [...messages].reverse().find((message) => message.role === "assistant");
  const recentAgentEvents = agentEvents
    .filter((event) => ["log", "tool_start", "tool_result", "file_delta", "error"].includes(event.event))
    .slice(-6);
  const [expanded, setExpanded] = useState(false);

  function setQuickPrompt(value) {
    setPrompt(value);
    window.requestAnimationFrame(() => {
      promptFrameRef.current?.querySelector("textarea")?.focus();
    });
  }

  return (
    <motion.aside
      className={collapsed ? "ai-assistance-panel collapsed" : "ai-assistance-panel"}
      aria-label="AI assistance"
      initial={{ x: 90, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5, delay: 0.18, ease: [0.22, 1, 0.36, 1] }}
    >
      {collapsed ? (
        <button className="assistant-collapsed-button" type="button" onClick={onToggleCollapsed} aria-label="Expand chat panel">
          <MessageSquare size={16} />
          <span>Chat</span>
        </button>
      ) : (
        <>
      <header className="assistant-head">
        <strong>AI Assistance</strong>
        <button type="button" onClick={onToggleCollapsed} aria-label="Collapse chat panel">
          <PanelLeftClose size={14} />
          Chat
        </button>
      </header>

      <div className="assistant-scroll">
        <article className="assistant-card primary">
          <p>
            {latestUser?.content ??
              "Create a modern, responsive landing page for StreamLine using HTML + TailwindCSS CDN. Deliver a single HTML file."}
          </p>
          <div className="assistant-section-list">
            <span>Sections:</span>
            <small>top-nav - sticky nav, logo + links + CTA</small>
            <small>hero - headline, subcopy, 2 CTAs, gradient bg + image</small>
          </div>
          {expanded && (
            <div className="assistant-section-list extended">
              <small>sections - feature cards, proof row, responsive footer</small>
              <small>quality - semantic HTML, accessible controls, clean export</small>
            </div>
          )}
          <button className="see-more" type="button" onClick={() => setExpanded((open) => !open)}>
            {expanded ? "See less" : "See more"} <ChevronDown size={13} />
          </button>
        </article>

        <p className="assistant-copy">
          I can help you understand, improve, fix, and test your code. What would you like me to help you with?
        </p>

        <div className="assistant-actions">
          <button type="button" onClick={() => setQuickPrompt(`Explain the current ${activeFile} file clearly.`)}>
            Explain this code
          </button>
          <button type="button" onClick={() => setQuickPrompt("Optimize the current website for performance and production deployment.")}>
            Optimize performance
          </button>
          <button type="button" onClick={() => setQuickPrompt("Find and fix likely errors in the current generated files.")}>
            Error fixing
          </button>
        </div>

        <article className="assistant-card checklist">
          <div className="assistant-card-title">
            <Bot size={15} />
            <strong>Analyzing {activeFile}</strong>
          </div>
          <ul>
            <li>Generated files are linked in Explorer</li>
            <li>Preview mode is available in the top switch</li>
          </ul>
        </article>

        {(isAgentStreaming || recentAgentEvents.length > 0) && (
          <article className="assistant-card agent-timeline-card">
            <div className="assistant-card-title">
              <Sparkles size={15} />
              <strong>{isAgentStreaming ? "Agent running" : "Agent timeline"}</strong>
            </div>
            <ul>
              {recentAgentEvents.map((event) => (
                <li key={event.id}>
                  <span>{event.event.replace("_", " ")}</span>
                  <small>
                    {String(
                      event.data.message ??
                        event.data.tool ??
                        event.data.path ??
                        event.data.stage ??
                        "Workflow event"
                    )}
                  </small>
                </li>
              ))}
            </ul>
          </article>
        )}
      </div>

      <FollowUpPrompt
        value={prompt}
        setValue={setPrompt}
        onSubmit={onSubmit}
        busy={busy}
        promptFrameRef={promptFrameRef}
      />
        </>
      )}
    </motion.aside>
  );
}

function BillingModal({ onClose, user, profile, onNotify }) {
  const { openAuth } = useAuthModal();
  const plans = [
    {
      name: "Pro",
      accent: "pro",
      icon: Blocks,
      monthlyPrice: 35,
      credits: "3,000",
      audience: "For professionals and small teams",
      features: ["Unlimited libraries", "Unlimited models", "Up to 25 editors"],
    },
    {
      name: "Studio",
      accent: "studio",
      icon: Bot,
      monthlyPrice: 75,
      credits: "8,000",
      audience: "For larger teams and studios",
      features: ["Unlimited storage", "Personal support", "Opt out of data training", "Unlimited editors"],
    },
  ];
  const [selectedPlanName, setSelectedPlanName] = useState("Pro");
  const [billingCycle, setBillingCycle] = useState("monthly");
  const selectedPlan = plans.find((plan) => plan.name === selectedPlanName) ?? plans[0];
  const PlanIcon = selectedPlan.icon;
  const userName = displayNameFromProfile(profile, user);
  const teamName = user ? `${userName}'s Team` : "MINDIGENOUS Team";
  const annualTotal = Math.round(selectedPlan.monthlyPrice * 12 * 0.8);
  const checkoutAmount = billingCycle === "monthly" ? selectedPlan.monthlyPrice : annualTotal;
  const billingCopy = billingCycle === "monthly" ? "billed monthly" : "billed annually";

  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape") onClose();
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  function handleCheckout(plan) {
    if (!user) {
      onClose();
      openAuth("signin");
      return;
    }

    onNotify?.(`${plan.name} checkout selected`);
  }

  return (
    <motion.div
      className="billing-modal-backdrop"
      onClick={onClose}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.18, ease: "easeOut" }}
    >
      <motion.section
        className={`billing-modal ${selectedPlan.accent}`}
        aria-label="Upgrade plan"
        onClick={(event) => event.stopPropagation()}
        initial={{ y: 18, scale: 0.97, opacity: 0 }}
        animate={{ y: 0, scale: 1, opacity: 1 }}
        exit={{ y: 10, scale: 0.98, opacity: 0 }}
        transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
      >
        <button
          type="button"
          className="billing-close"
          onClick={(event) => {
            event.stopPropagation();
            onClose();
          }}
          aria-label="Close billing"
        >
          <X size={15} />
        </button>

        <aside className="billing-plan-preview" aria-label={`${selectedPlan.name} plan summary`}>
          <p>You are currently on the <strong>{profile?.plan ?? "free"} plan</strong></p>
          <article>
            <span className="billing-plan-icon"><PlanIcon size={20} /></span>
            <strong>{selectedPlan.name}</strong>
            <div className="billing-price">
              <span>€{selectedPlan.monthlyPrice}.00</span>
              <small>per editor/month<br />{billingCopy}</small>
            </div>
            <p>{selectedPlan.audience}</p>
            <button type="button" className="billing-credit-select">
              {selectedPlan.credits} AI credits per month <ChevronDown size={13} />
            </button>
            <ul>
              {selectedPlan.features.map((feature) => (
                <li key={feature}>
                  <CheckCircle2 size={15} />
                  {feature}
                </li>
              ))}
            </ul>
          </article>
        </aside>

        <div className="billing-checkout">
          <header className="billing-modal-head">
            <span className="billing-team-avatar">{teamName.slice(0, 1).toUpperCase()}</span>
            <small>{teamName}</small>
          </header>

          <div className="billing-title-row">
            <h2>
              Upgrade to MINDIGENOUS <span>{selectedPlan.name}</span>
            </h2>
            <div className="billing-plan-tabs" aria-label="Choose plan">
              {plans.map((plan) => (
                <button
                  type="button"
                  key={plan.name}
                  className={selectedPlan.name === plan.name ? "active" : ""}
                  onClick={() => setSelectedPlanName(plan.name)}
                >
                  {plan.name}
                </button>
              ))}
            </div>
          </div>

          <section className="billing-cycle-row" aria-label="Billing cycle">
            <span>Billing cycle</span>
            <div>
              <button
                type="button"
                className={billingCycle === "monthly" ? "active" : ""}
                onClick={() => setBillingCycle("monthly")}
              >
                Monthly
              </button>
              <button
                type="button"
                className={billingCycle === "annual" ? "active" : ""}
                onClick={() => setBillingCycle("annual")}
              >
                Annually <em>-20%</em>
              </button>
            </div>
          </section>

          <section className="billing-summary">
            <div className="billing-summary-line">
              <span>
                <strong>MINDIGENOUS {selectedPlan.name}</strong>
                <small>€{selectedPlan.monthlyPrice}.00 × 1 editor</small>
              </span>
              <strong>€{checkoutAmount}</strong>
            </div>
            <article className="billing-start-card">
              <CheckCircle2 size={16} />
              <div>
                <strong>Start your subscription</strong>
                <p>
                  You will get full access to all {selectedPlan.name} features. Each editor receives {selectedPlan.credits} AI credits per month.
                </p>
              </div>
            </article>
            <div className="billing-promo-row">
              <input type="text" placeholder="Promotion code" aria-label="Promotion code" />
              <button type="button">Apply</button>
            </div>
            <button type="button" className="billing-submit" onClick={() => handleCheckout(selectedPlan)}>
              Start subscription
            </button>
          </section>
        </div>

      </motion.section>
    </motion.div>
  );
}

function CommandSearch({ files, query, setQuery, onOpenFile, onClose }) {
  const allFiles = useMemo(() => {
    const names = new Set([
      ...Object.keys(workspaceTemplates),
      ...Object.keys(files),
      "package.json",
      "README.md",
      ".gitignore",
    ]);

    return Array.from(names);
  }, [files]);

  const results = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return allFiles.slice(0, 8);

    return allFiles
      .filter((file) => {
        const content = files[file] ?? workspaceTemplates[file] ?? "";
        return file.toLowerCase().includes(needle) || content.toLowerCase().includes(needle);
      })
      .slice(0, 10);
  }, [allFiles, files, query]);

  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape") onClose();
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <motion.div
      className="command-search-backdrop"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.16 }}
      onMouseDown={onClose}
    >
      <motion.div
        className="command-search"
        initial={{ y: -18, scale: 0.98 }}
        animate={{ y: 0, scale: 1 }}
        exit={{ y: -18, scale: 0.98 }}
        transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="command-search-input">
          <Search size={16} />
          <input
            autoFocus
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search files, symbols, and generated code"
          />
          <kbd>Esc</kbd>
        </div>
        <div className="command-results">
          {results.map((file) => (
            <button
              type="button"
              key={file}
              onClick={() => {
                onOpenFile(file);
                onClose();
              }}
            >
              <FileIcon file={file} />
              <span>{file}</span>
              <small>{fileKind(file)}</small>
            </button>
          ))}
          {!results.length && <p>No results found.</p>}
        </div>
      </motion.div>
    </motion.div>
  );
}

function WorkspaceMode({
  prompt,
  setPrompt,
  onSubmit,
  busy,
  messages,
  agentEvents,
  isAgentStreaming,
  files,
  setFiles,
  activeFile,
  setActiveFile,
  history,
  onRun,
  onExport,
  onDeploy,
  onHome,
  onInvite,
  onManageAccess,
  onNotify,
  onCopyCode,
  refreshKey,
  promptFrameRef,
  designSettings,
  onDesignSettingsChange,
  projectName,
  setProjectName,
  projects,
  onOpenProject,
  onRenameProject,
  user,
}) {
  const [panelsReady, setPanelsReady] = useState(false);
  const [editorView, setEditorView] = useState("code");
  const [openTabs, setOpenTabs] = useState(["index.html", "styles.css", "script.js"]);
  const [zoom, setZoom] = useState(88);
  const [fullscreen, setFullscreen] = useState(false);
  const [activeTool, setActiveTool] = useState("explorer");
  const [projectMenuOpen, setProjectMenuOpen] = useState(false);
  const [customFolders, setCustomFolders] = useState([]);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [chatCollapsed, setChatCollapsed] = useState(false);
  const [terminalOpen, setTerminalOpen] = useState(false);
  const [settingsModalOpen, setSettingsModalOpen] = useState(false);

  useEffect(() => {
    const timer = window.setTimeout(() => setPanelsReady(true), 220);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!files[activeFile]) return;
    if (!fileOrder.includes(activeFile)) {
      setActiveFile("index.html");
      return;
    }
    setOpenTabs((current) => (
      current.includes(activeFile)
        ? current.filter((file) => fileOrder.includes(file) && files[file])
        : [activeFile, ...current.filter((file) => fileOrder.includes(file) && files[file])].slice(0, 6)
    ));
  }, [activeFile, files, setActiveFile]);

  function openFile(file) {
    setFiles((current) => (
      current[file] ? current : { ...current, [file]: templateForFile(file) }
    ));
    setActiveFile(file);
    setEditorView("code");
    setOpenTabs((current) => (
      current.includes(file) ? current : [file, ...current].slice(0, 6)
    ));
  }

  function closeTab(file) {
    setOpenTabs((current) => {
      const next = current.filter((item) => item !== file);
      if (activeFile === file) {
        const fallback = next.find((item) => files[item]) ?? Object.keys(files)[0];
        if (fallback) setActiveFile(fallback);
      }
      return next.length ? next : [Object.keys(files)[0] ?? "index.html"];
    });
  }

  function createFile() {
    const count = Object.keys(files).filter((file) => file.startsWith("component-")).length + 1;
    const file = `component-${count}.tsx`;
    setFiles((current) => ({ ...current, [file]: templateForFile(file) }));
    openFile(file);
    onNotify?.(`Created ${file}`);
  }

  function createFolder() {
    const count = customFolders.length + 1;
    const folder = `folder-${count}`;
    setCustomFolders((current) => [...current, folder]);
    onNotify?.(`Created ${folder}`);
  }

  function handleToolChange(tool) {
    if (tool === "settings") {
      setSettingsModalOpen((open) => !open);
      onNotify?.(settingsModalOpen ? "Settings closed" : "Settings opened");
      return;
    }

    setSettingsModalOpen(false);
    setActiveTool((current) => (current === tool ? null : tool));
    if (tool === "extensions") onNotify?.("Extensions panel selected");
    if (tool === "repository") onNotify?.("Repository panel selected");
  }

  function handleZoomChange() {
    setZoom((current) => (current >= 112 ? 76 : current + 12));
  }

  function selectProject(project) {
    setProjectMenuOpen(false);
    onOpenProject?.(project);
    onNotify?.(`Opened ${project.name}`);
  }

  function renameProject(project) {
    setProjectName(project);
    onRenameProject?.(project);
    onNotify?.(`Renamed project to ${project}`);
  }

  const workspaceBodyClassName = [
    "workspace-body",
    chatCollapsed ? "chat-collapsed" : "",
    (!activeTool || settingsModalOpen) ? "explorer-collapsed" : "",
    editorView === "design" ? "design-focused" : "",
  ].filter(Boolean).join(" ");

  return (
    <section className="workspace-mode" aria-label="MINDI workspace">
      <main className="workspace-main">
        {panelsReady && (
          <div className="reference-workspace-shell">
            <WorkspaceTopChrome
              editorView={editorView}
              setEditorView={setEditorView}
              onHome={onHome}
              projectName={projectName}
              projectMenuOpen={projectMenuOpen}
              setProjectMenuOpen={setProjectMenuOpen}
              onProjectSelect={selectProject}
              onProjectRename={renameProject}
              projects={projects}
              onInvite={onInvite}
              onManageAccess={onManageAccess}
              onNotify={onNotify}
            />
            <div className={workspaceBodyClassName}>
              <WorkspaceRail
                activeTool={settingsModalOpen ? "settings" : activeTool}
                onToolChange={handleToolChange}
                onHome={onHome}
                onRun={onRun}
                onExport={onExport}
                onDeploy={onDeploy}
              />
              {!settingsModalOpen && activeTool && (
                <ExplorerPanel
                  files={files}
                  activeFile={activeFile}
                  editorView={editorView}
                  history={history}
                  activeTool={activeTool}
                  onOpenFile={openFile}
                  terminalOpen={terminalOpen}
                  onOpenTerminal={() => {
                    setEditorView("code");
                    setTerminalOpen((open) => !open);
                  }}
                  onCreateFile={createFile}
                  onCreateFolder={createFolder}
                  onOpenSearch={() => {
                    setSearchQuery("");
                    setSearchOpen(true);
                  }}
                  onNotify={onNotify}
                  customFolders={customFolders}
                />
              )}
              <CodeWorkspace
                files={files}
                setFiles={setFiles}
                activeFile={activeFile}
                setActiveFile={setActiveFile}
                editorView={editorView}
                setEditorView={setEditorView}
                refreshKey={refreshKey}
                openTabs={openTabs}
                onOpenFile={openFile}
                onCloseTab={closeTab}
                onCreateFile={createFile}
                onCopyCode={onCopyCode}
                zoom={zoom}
                onZoomChange={handleZoomChange}
                fullscreen={fullscreen}
                onToggleFullscreen={() => setFullscreen((value) => !value)}
                terminalOpen={terminalOpen}
                designSettings={designSettings}
                onDesignSettingsChange={onDesignSettingsChange}
              />
              <AssistancePanel
                messages={messages}
                activeFile={activeFile}
                busy={busy}
                agentEvents={agentEvents}
                isAgentStreaming={isAgentStreaming}
                prompt={prompt}
                setPrompt={setPrompt}
                onSubmit={onSubmit}
                promptFrameRef={promptFrameRef}
                collapsed={chatCollapsed}
                onToggleCollapsed={() => setChatCollapsed((value) => !value)}
              />
              <AnimatePresence>
                {searchOpen && (
                  <CommandSearch
                    files={files}
                    query={searchQuery}
                    setQuery={setSearchQuery}
                    onOpenFile={openFile}
                    onClose={() => setSearchOpen(false)}
                  />
                )}
              </AnimatePresence>
              <AnimatePresence>
                {settingsModalOpen && (
                  <SettingsModal
                    onClose={() => setSettingsModalOpen(false)}
                    onNotify={onNotify}
                    user={user}
                  />
                )}
              </AnimatePresence>
            </div>
          </div>
        )}
      </main>
    </section>
  );
}

function defaultPromptFrame() {
  const width = Math.min(720, Math.max(320, window.innerWidth - 48));
  const height = 136;

  return {
    left: (window.innerWidth - width) / 2,
    top: Math.max(160, window.innerHeight / 2 - height / 2 + 70),
    width,
    height,
  };
}

function ModeTransitionFrame({ direction, promptFrame, viewport }) {
  const fallbackFrame = defaultPromptFrame();
  const prompt = promptFrame ?? fallbackFrame;
  const inset = Math.min(22, Math.max(14, viewport.width * 0.012));
  const full = {
    left: inset,
    top: inset,
    width: Math.max(0, viewport.width - inset * 2),
    height: Math.max(0, viewport.height - inset * 2),
  };
  const from = direction === "enter" ? prompt : full;
  const to = direction === "enter" ? full : prompt;
  const fromRadius = direction === "enter" ? 22 : 36;
  const toRadius = direction === "enter" ? 36 : 22;

  return (
    <motion.div
      className="mode-transition-frame"
      initial={{
        left: from.left,
        top: from.top,
        width: from.width,
        height: from.height,
        borderRadius: fromRadius,
        opacity: 1,
      }}
      animate={{
        left: to.left,
        top: to.top,
        width: to.width,
        height: to.height,
        borderRadius: toRadius,
        opacity: 1,
      }}
      exit={{ opacity: 0 }}
      transition={frameTransition}
    />
  );
}

export default function App() {
  const { user, isConfigured } = useAuthSession();
  const agentStream = useAgentStream();
  const [mode, setMode] = useState("chat");
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [transitionDirection, setTransitionDirection] = useState(null);
  const [uiSuppressed, setUiSuppressed] = useState(false);
  const [storedPromptFrame, setStoredPromptFrame] = useState(null);
  const [viewport, setViewport] = useState({
    width: typeof window === "undefined" ? 1440 : window.innerWidth,
    height: typeof window === "undefined" ? 900 : window.innerHeight,
  });
  const promptFrameRef = useRef(null);
  const transitionTimers = useRef([]);
  const streamingAssistantMessageId = useRef(null);
  const [messages, setMessages] = useState([]);
  const [designSettings, setDesignSettings] = useState(() => readStoredDesignSettings());
  const [files, setFiles] = useState(() => applyDesignToFiles(starterFiles, readStoredDesignSettings()));
  const [activeFile, setActiveFile] = useState("index.html");
  const [history, setHistory] = useState([]);
  const [profile, setProfile] = useState(() => profileFromUser(user));
  const [cloudProjects, setCloudProjects] = useState([]);
  const [cloudLoading, setCloudLoading] = useState(false);
  const [currentProjectId, setCurrentProjectId] = useState(null);
  const [projectName, setProjectName] = useState("Untitled");
  const [refreshKey, setRefreshKey] = useState(0);
  const [toast, setToast] = useState("");
  const [accessModalOpen, setAccessModalOpen] = useState(false);
  const [billingOpen, setBillingOpen] = useState(false);

  const currentPreview = useMemo(() => composePreview(files), [files]);

  useEffect(() => {
    const cloudAppearance = profile?.appearance_settings;
    const hasCloudAppearance = cloudAppearance && Object.keys(cloudAppearance).length > 0;
    const nextSettings = hasCloudAppearance
      ? normalizeAppearanceSettings(cloudAppearance)
      : readStoredAppearanceSettings(user);

    applyAppearanceSettings(nextSettings);
    if (hasCloudAppearance) {
      try {
        writeStoredAppearanceSettings(nextSettings, user);
      } catch {
        // Cloud preferences remain authoritative even when local storage is unavailable.
      }
    }
  }, [profile?.appearance_settings, user?.email, user?.id]);

  useEffect(() => {
    function handleResize() {
      setViewport({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    }

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    return () => {
      transitionTimers.current.forEach((timer) => window.clearTimeout(timer));
    };
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(DESIGN_SETTINGS_STORAGE_KEY, JSON.stringify(designSettings));
    } catch {
      // Local persistence is a convenience; the in-memory design state remains authoritative.
    }
  }, [designSettings]);

  useEffect(() => {
    if (mode !== "workspace" || !user || !isConfigured || !currentProjectId) return undefined;

    const timer = window.setTimeout(() => {
      persistCurrentProject(files, designSettings);
    }, 900);

    return () => window.clearTimeout(timer);
  }, [currentProjectId, designSettings, files, isConfigured, mode, projectName, user]);

  useEffect(() => {
    let cancelled = false;

    async function loadCloudData() {
      if (!user || !isConfigured) {
        setProfile(profileFromUser(user));
        setCloudProjects([]);
        setHistory([]);
        setCloudLoading(false);
        return;
      }

      setCloudLoading(true);
      const [nextProfile, nextProjects] = await Promise.all([
        fetchCurrentProfile(user),
        listUserProjects(user.id),
      ]);

      if (cancelled) return;
      setProfile(nextProfile);
      setCloudProjects(nextProjects);
      setHistory(nextProjects.map((project) => project.name));
      setCloudLoading(false);
    }

    loadCloudData();

    return () => {
      cancelled = true;
    };
  }, [isConfigured, user]);

  function queueTransitionTimer(callback, delay) {
    const timer = window.setTimeout(callback, delay);
    transitionTimers.current.push(timer);
  }

  function getPromptFrame() {
    const rect = promptFrameRef.current?.getBoundingClientRect();

    if (!rect) return defaultPromptFrame();

    return {
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height,
    };
  }

  function startWorkspaceTransition({ releaseBusy = true } = {}) {
    const frame = getPromptFrame();
    setStoredPromptFrame(frame);
    setTransitionDirection("enter");
    setUiSuppressed(true);

    queueTransitionTimer(() => {
      setMode("workspace");
    }, 720);

    queueTransitionTimer(() => {
      setTransitionDirection(null);
      setUiSuppressed(false);
      if (releaseBusy) setBusy(false);
    }, 980);
  }

  function startHomeTransition() {
    setTransitionDirection("exit");
    setUiSuppressed(true);

    queueTransitionTimer(() => {
      setMode("chat");
    }, 700);

    queueTransitionTimer(() => {
      setTransitionDirection(null);
      setUiSuppressed(false);
    }, 880);
  }

  function addMessage(role, content) {
    setMessages((current) => [
      ...current,
      {
        id: `${role}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        role,
        content,
      },
    ]);
  }

  function beginStreamingAssistant() {
    const id = `assistant-stream-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    streamingAssistantMessageId.current = id;
    setMessages((current) => [...current, { id, role: "assistant", content: "", streaming: true }]);
  }

  function appendStreamingAssistant(text) {
    const id = streamingAssistantMessageId.current;
    if (!id) beginStreamingAssistant();
    const activeId = streamingAssistantMessageId.current;
    setMessages((current) => current.map((message) => (
      message.id === activeId
        ? { ...message, content: `${message.content}${text}`, streaming: true }
        : message
    )));
  }

  function finishStreamingAssistant(fallback) {
    const id = streamingAssistantMessageId.current;
    if (!id) {
      if (fallback) addMessage("assistant", fallback);
      return;
    }
    setMessages((current) => current.map((message) => (
      message.id === id
        ? { ...message, content: message.content.trim() || fallback, streaming: false }
        : message
    )));
    streamingAssistantMessageId.current = null;
  }

  function mergeCloudProject(project) {
    if (!project) return;
    setCloudProjects((current) => [project, ...current.filter((item) => item.id !== project.id)]);
    setHistory((current) => [project.name, ...current.filter((item) => item !== project.name)].slice(0, 12));
  }

  async function persistCurrentProject(nextFiles = files, nextDesignSettings = designSettings, sourcePrompt) {
    if (!user || !isConfigured) return null;

    const saved = await saveProjectToCloud({
      id: currentProjectId,
      ownerId: user.id,
      name: projectName,
      files: nextFiles,
      designSettings: nextDesignSettings,
      sourcePrompt,
    });
    mergeCloudProject(saved);
    if (saved?.id) setCurrentProjectId(saved.id);
    return saved;
  }

  async function openCloudProject(project) {
    if (!project) return;

    const nextDesignSettings = normalizeDesignSettings(project.design_settings ?? {});
    const nextFiles = project.files && Object.keys(project.files).length
      ? project.files
      : applyDesignToFiles(starterFiles, nextDesignSettings);

    setCurrentProjectId(project.id);
    setProjectName(project.name);
    setDesignSettings(nextDesignSettings);
    setFiles(nextFiles);
    setActiveFile(nextFiles["index.html"] ? "index.html" : Object.keys(nextFiles)[0] ?? "index.html");
    setRefreshKey((key) => key + 1);
    setHistory((current) => [project.name, ...current.filter((item) => item !== project.name)].slice(0, 12));
    setMode("workspace");
    setTransitionDirection(null);
    setUiSuppressed(false);
    touchCloudProject(project.id);
  }

  function createFromPromptLocal(value) {
    const nextFiles = applyDesignToFiles(generateFiles(value), designSettings);
    const title = titleFromPrompt(value);
    setProjectName(title);
    setFiles(nextFiles);
    setActiveFile("index.html");
    setHistory((current) => [title, ...current.filter((item) => item !== title)].slice(0, 6));
    setRefreshKey((key) => key + 1);
    addMessage("assistant", `Generated ${title}. Code and preview are ready.`);
    if (user && isConfigured) {
      saveProjectToCloud({
        id: mode === "workspace" ? currentProjectId : null,
        ownerId: user.id,
        name: title,
        files: nextFiles,
        designSettings,
        sourcePrompt: value,
      }).then((saved) => {
        mergeCloudProject(saved);
        if (saved?.id) setCurrentProjectId(saved.id);
      });
    }
  }

  async function createFromPrompt(value, { openWorkspace = false } = {}) {
    const title = titleFromPrompt(value);
    const startingFiles = openWorkspace ? applyDesignToFiles(starterFiles, designSettings) : files;
    const streamedFiles = {};

    setProjectName(title);
    if (openWorkspace) {
      setCurrentProjectId(null);
      setFiles(startingFiles);
      setActiveFile("index.html");
      setHistory((current) => [title, ...current.filter((item) => item !== title)].slice(0, 12));
      startWorkspaceTransition({ releaseBusy: false });
    }

    beginStreamingAssistant();

    try {
      await agentStream.startWorkflow(
        {
          prompt: value,
          project_id: openWorkspace ? null : currentProjectId,
          files: startingFiles,
          design_settings: designSettings,
          mode: openWorkspace ? "chat" : "workspace",
        },
        {
          onToken: appendStreamingAssistant,
          onFileDelta: (delta) => {
            const path = String(delta.path ?? "");
            if (!path) return;

            if (delta.operation === "delete") {
              delete streamedFiles[path];
              setFiles((current) => {
                const next = { ...current };
                delete next[path];
                return next;
              });
              return;
            }

            const content = String(delta.content ?? "");
            streamedFiles[path] = content;
            setFiles((current) => ({ ...current, [path]: content }));
            if (path === "index.html") setActiveFile("index.html");
            setRefreshKey((key) => key + 1);
          },
          onDone: (data) => {
            const summary = String(data.summary ?? `Generated ${title}. Code and preview are ready.`);
            finishStreamingAssistant(summary);
          },
        }
      );

      const finalFiles = Object.keys(streamedFiles).length
        ? { ...startingFiles, ...streamedFiles }
        : applyDesignToFiles(generateFiles(value), designSettings);
      setFiles(finalFiles);
      setActiveFile(finalFiles["index.html"] ? "index.html" : Object.keys(finalFiles)[0] ?? "index.html");
      setHistory((current) => [title, ...current.filter((item) => item !== title)].slice(0, 12));
      setRefreshKey((key) => key + 1);

      if (user && isConfigured) {
        saveProjectToCloud({
          id: openWorkspace ? null : currentProjectId,
          ownerId: user.id,
          name: title,
          files: finalFiles,
          designSettings,
          sourcePrompt: value,
        }).then((saved) => {
          mergeCloudProject(saved);
          if (saved?.id) setCurrentProjectId(saved.id);
        });
      }
    } catch (error) {
      const failedStreamId = streamingAssistantMessageId.current;
      if (failedStreamId) {
        setMessages((current) => current.filter((message) => message.id !== failedStreamId));
      }
      streamingAssistantMessageId.current = null;
      showToast("Backend unavailable. Using local generator.");
      createFromPromptLocal(value);
    } finally {
      setBusy(false);
    }
  }

  async function handleSubmit(event) {
    event?.preventDefault?.();
    const value = prompt.trim();
    if (!value || busy) return;

    setPrompt("");
    setBusy(true);
    addMessage("user", value);

    if (mode === "workspace") {
      if (isBuildIntent(value) && !isVagueBuild(value)) {
        await createFromPrompt(value);
      } else if (isVagueBuild(value)) {
        addMessage("assistant", "Tell me the page, audience, and visual style you want.");
        window.setTimeout(() => {
          setBusy(false);
        }, 360);
      } else {
        addMessage("assistant", `Added that context to the workspace. I can apply it to ${activeFile} when you send a build request.`);
        window.setTimeout(() => {
          setBusy(false);
        }, 360);
      }
      return;
    }

    if (isBuildIntent(value)) {
      if (isVagueBuild(value)) {
        addMessage("assistant", "What should I build?");
        window.setTimeout(() => {
          setBusy(false);
        }, 360);
      } else {
        await createFromPrompt(value, { openWorkspace: true });
      }
    } else {
      addMessage("assistant", "Ask me a question, or give me something to build.");
      window.setTimeout(() => {
        setBusy(false);
      }, 360);
    }
  }

  function showToast(value) {
    setToast(value);
    window.setTimeout(() => setToast(""), 1800);
  }

  function handleDesignSettingsChange(nextSettings) {
    const normalizedSettings = normalizeDesignSettings(nextSettings);
    setDesignSettings(normalizedSettings);
    setFiles((current) => applyDesignToFiles(current, normalizedSettings));
  }

  async function handleProjectRename(nextName) {
    if (!currentProjectId || !user || !isConfigured) return;
    const renamed = await renameCloudProject(currentProjectId, nextName);
    mergeCloudProject(renamed);
  }

  function handleRun() {
    setRefreshKey((key) => key + 1);
    showToast("Preview refreshed");
  }

  function handleExport() {
    const blob = new Blob([currentPreview], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "mindi-export.html";
    link.click();
    URL.revokeObjectURL(url);
    showToast("Exported HTML");
  }

  function handleDeploy() {
    addMessage("assistant", "Deployment package prepared.");
    showToast("Deploy draft ready");
  }

  function handleInvite() {
    navigator.clipboard?.writeText("https://mindigenous.local/invite").catch(() => {});
    showToast("Invite link copied");
  }

  function handleCopyCode(value) {
    const fallbackCopy = () => {
      const textarea = document.createElement("textarea");
      textarea.value = value;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      const copied = document.execCommand("copy");
      document.body.removeChild(textarea);
      showToast(copied ? "Copied code" : "Copy unavailable");
    };

    if (!navigator.clipboard?.writeText) {
      fallbackCopy();
      return;
    }

    navigator.clipboard.writeText(value).then(
      () => showToast("Copied code"),
      fallbackCopy
    );
  }

  return (
    <div
      className={`app ${mode === "workspace" ? "is-workspace" : "is-chat"} ${
        uiSuppressed ? "is-transitioning" : ""
      }`}
    >
      <LetterGlitchBackground centerVignette />
      <LayoutGroup id="mindi-morph">
        {mode === "chat" ? (
          <ChatMode
            prompt={prompt}
            setPrompt={setPrompt}
            onSubmit={handleSubmit}
            busy={busy}
            promptFrameRef={promptFrameRef}
            onNotify={showToast}
            user={user}
            profile={profile}
            projects={cloudProjects}
            cloudLoading={cloudLoading}
            onOpenProject={openCloudProject}
            onOpenUpgrade={() => setBillingOpen(true)}
          />
        ) : (
          <WorkspaceMode
            prompt={prompt}
            setPrompt={setPrompt}
            onSubmit={handleSubmit}
            busy={busy}
            messages={messages}
            agentEvents={agentStream.events}
            isAgentStreaming={agentStream.isStreaming}
            files={files}
            setFiles={setFiles}
            activeFile={activeFile}
            setActiveFile={setActiveFile}
            history={history}
            onRun={handleRun}
            onExport={handleExport}
            onDeploy={handleDeploy}
            onHome={startHomeTransition}
            onInvite={handleInvite}
            onManageAccess={() => setAccessModalOpen(true)}
            onNotify={showToast}
            onCopyCode={handleCopyCode}
            refreshKey={refreshKey}
            promptFrameRef={promptFrameRef}
            designSettings={designSettings}
            onDesignSettingsChange={handleDesignSettingsChange}
            projectName={projectName}
            setProjectName={setProjectName}
            projects={cloudProjects}
            onOpenProject={openCloudProject}
            onRenameProject={handleProjectRename}
            user={user}
          />
        )}
      </LayoutGroup>

      <AnimatePresence>
        {transitionDirection && (
          <ModeTransitionFrame
            key={transitionDirection}
            direction={transitionDirection}
            promptFrame={storedPromptFrame}
            viewport={viewport}
          />
        )}
      </AnimatePresence>

      <AuthModal />
      <AnimatePresence>
        {accessModalOpen && (
          <ManageAccessModal
            onClose={() => setAccessModalOpen(false)}
            onCopyInvite={handleInvite}
            onNotify={showToast}
          />
        )}
      </AnimatePresence>
      <AnimatePresence>
        {billingOpen && (
          <BillingModal
            onClose={() => setBillingOpen(false)}
            user={user}
            profile={profile}
            onNotify={showToast}
          />
        )}
      </AnimatePresence>
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
