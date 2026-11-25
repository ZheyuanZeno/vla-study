'use strict';

const byId = (id) => document.getElementById(id);
const play = byId('play');
const seek = byId('seek');
const stepInput = byId('step');
const controls = [play, seek, stepInput, byId('prev'), byId('next')];
let players;
let master;
let maxStep;
let fps;
const labels = {
  Baseline: 'Baseline', T1: 'T1 · Small', T2: 'T2 · Medium', T3: 'T3 · Large',
  O1: 'O1 · Small', O2: 'O2 · Medium', O3: 'O3 · Large',
  OB1: 'OB1 · Small', OB2: 'OB2 · Medium', OB3: 'OB3 · Large',
  B1: 'B1 · Small', B2: 'B2 · Medium', B3: 'B3 · Large',
};

function buildPlayers(state) {
  const cases = caseSets[state];
  for (const section of ['baseline', 'target', 'background', 'object', 'object-background', 'initial']) {
    byId(section).replaceChildren();
  }
  byId("object-family").hidden = !cases.some(item => item.condition === "O1");
  maxStep = Math.max(...cases.map((item) => item.steps));
  fps = cases[0].fps;
  seek.max = stepInput.max = maxStep;
  players = cases.map((item) => {
    const card = document.createElement('article');
    card.className = 'frame';
    const heading = `${labels[item.condition]} · ${item.success ? 'Success' : 'Timeout'}`;
    card.innerHTML = `<div class="label">${heading}</div>
      <div class="media"><video preload="metadata" muted playsinline aria-label="${labels[item.condition]} rollout"></video>
        <div class="outline" hidden></div></div>
      <div class="caption"></div><div class="links"></div>`;
    const box = card.querySelector('.outline');
    if (item.bbox) {
      const [x0, y0, x1, y1] = item.bbox;
      const percent = (pixel) => `${pixel * 100 / 512}%`;
      Object.assign(box.style, {
        left: percent(x0), top: percent(y0), width: percent(x1 - x0), height: percent(y1 - y0),
      });
    }
    const links = card.querySelector('.links');
    links.innerHTML = `<a href="${item.original}">Original MP4</a>`;
    if (item.reconstruction) {
      links.insertAdjacentHTML('beforeend', `<a href="${item.reconstruction}">Reconstruction MP4</a>`);
    }
    const destination = item.condition === 'Baseline' ? 'baseline'
      : item.condition.startsWith('OB') ? 'object-background'
      : item.condition.startsWith('O') ? 'object'
      : item.condition.startsWith('T') ? 'target' : 'background';
    byId(destination).append(card);
    const size = item.side ? ` · ${item.side} × ${item.side} px` : '';
    const coverage = item.side ? ` · ${(item.coverage * 100).toFixed(1)}% ${item.condition.startsWith('O') ? 'bowl' : 'plate'} coverage` : '';
    byId('initial').insertAdjacentHTML('beforeend', `<figure class="frame">
      <a href="${item.actual}"><img loading="lazy" width="512" height="512" src="${item.actual}" alt="${labels[item.condition]} saved first model input"></a>
      <figcaption class="caption">${labels[item.condition]}${size}${coverage}</figcaption>
    </figure>`);
    return {item, video: card.querySelector('video'), box, caption: card.querySelector('.caption')};
  });
  master = players.find(({item}) => item.steps === maxStep).video;
}

let step = 0;
let mode = 'reconstruction';
let running = false;
let ready = false;
let animation = 0;
let sourceRevision = 0;

function displayStep(value) {
  step = value;
  seek.value = step;
  if (document.activeElement !== stepInput) stepInput.value = step;
  for (const {item, caption} of players) {
    const label = mode === 'reconstruction' && item.bbox
      ? 'Mask overlay reconstruction'
      : mode === 'outline' && item.bbox ? 'Original · mask outline' : 'Original';
    caption.textContent = `${label} · Step ${Math.min(step, item.steps)} / ${item.steps}`
      + (step > item.steps ? ' · ended' : '');
  }
}

function placeVideos() {
  for (const {item, video} of players) {
    // Seek a quarter of a frame inside the intended sample, away from the frame boundary.
    video.currentTime = (Math.min(step, item.steps) + 0.25) / fps;
  }
}

function pause() {
  running = false;
  cancelAnimationFrame(animation);
  players.forEach(({video}) => video.pause());
  play.textContent = step === maxStep ? 'Replay together' : 'Play together';
}

function jump(value) {
  if (!ready || !Number.isFinite(value)) return;
  pause();
  displayStep(Math.max(0, Math.min(maxStep, Math.round(value))));
  stepInput.value = step;
  placeVideos();
  play.textContent = step === maxStep ? 'Replay together' : 'Play together';
}

function tick() {
  if (!running) return;
  displayStep(Math.min(maxStep, Math.floor(master.currentTime * fps + 0.001)));
  for (const {item, video} of players) {
    if (video === master) continue;
    if (step >= item.steps && (video.currentTime * fps < item.steps || !video.paused)) {
      video.pause();
      video.currentTime = (item.steps + 0.25) / fps;
    } else if (step < item.steps && !video.seeking && Math.abs(video.currentTime - master.currentTime) > 1.5 / fps) {
      video.currentTime = master.currentTime;
    }
  }
  if (master.ended) {
    jump(maxStep);
    return;
  }
  animation = requestAnimationFrame(tick);
}

play.addEventListener('click', async () => {
  if (running) {
    pause();
    placeVideos();
    return;
  }
  if (!ready) return;
  if (step === maxStep) jump(0);
  running = true;
  play.textContent = 'Pause';
  try {
    await Promise.all(players.filter(({item}) => step < item.steps).map(({video}) => video.play()));
    if (running) tick();
  } catch (error) {
    if (!running || error.name === 'AbortError') return;
    pause();
    byId('status').textContent = 'Playback could not start. Use the step controls or open an individual MP4.';
  }
});
seek.addEventListener('input', () => jump(seek.valueAsNumber));
stepInput.addEventListener('change', () => {
  jump(stepInput.valueAsNumber);
  stepInput.value = step;
});
byId('prev').addEventListener('click', () => jump(step - 1));
byId('next').addEventListener('click', () => jump(step + 1));
byId('modes').addEventListener('change', (event) => loadMode(event.target.value));

function loadMode(value) {
  mode = value;
  const revision = ++sourceRevision;
  pause();
  ready = false;
  controls.forEach((control) => { control.disabled = true; });
  byId('status').textContent = 'Loading videos…';
  byId('mode-note').textContent = {
    original: 'All conditions show the original external-camera recordings.',
    reconstruction: 'Mask overlay reconstruction. The gray square is fixed at the saved coordinates. Baseline remains unmasked.',
    outline: 'Orange outlines show the mask boundaries over the original frames.',
  }[mode];
  displayStep(step);
  const waiting = new Set(players);
  for (const player of players) {
    const {item, video, box} = player;
    const source = (mode === 'reconstruction' && item.reconstruction) || item.original;
    box.hidden = mode !== 'outline' || !item.bbox;
    const loaded = () => {
      if (revision !== sourceRevision) return;
      waiting.delete(player);
      if (waiting.size) return;
      placeVideos();
      ready = true;
      controls.forEach((control) => { control.disabled = false; });
      byId('status').textContent = 'Ready. View changes pause playback and retain the current step.';
    };
    video.onloadedmetadata = loaded;
    video.onerror = () => {
      if (revision !== sourceRevision) return;
      byId('status').textContent = 'One or more videos could not load. Try the individual MP4 links.';
    };
    if (video.getAttribute('src') !== source) video.src = source;
    else if (video.readyState >= HTMLMediaElement.HAVE_METADATA) loaded();
  }
}

byId('state').addEventListener('change', (event) => {
  pause();
  buildPlayers(event.target.value);
  step = 0;
  stepInput.value = 0;
  loadMode(mode);
});

buildPlayers(byId('state').value);
loadMode(mode);
