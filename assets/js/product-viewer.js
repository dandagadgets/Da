import * as THREE from 'three';
import { GLTFLoader } from './three-lib/GLTFLoader.js';
import { DRACOLoader } from './three-lib/DRACOLoader.js';
import { GLTFMaterialsPbrSpecularGlossinessExtension } from './three-lib/SpecGlossExtension.js';

var dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath('./assets/js/three-lib/draco/');

var loader = new GLTFLoader();
loader.setDRACOLoader(dracoLoader);
loader.register(function (parser) { return new GLTFMaterialsPbrSpecularGlossinessExtension(parser); });

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

function createViewer(pcard) {
  var canvas = pcard.querySelector('.pcard-canvas');
  var icon = pcard.querySelector('.pcard-icon');
  var loaderEl = pcard.querySelector('.pcard-loader');
  var modelUrl = pcard.getAttribute('data-model');
  if (!canvas || !modelUrl) return null;

  var renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true, powerPreference: 'high-performance' });
  } catch (e) {
    return null;
  }
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(35, 1, 0.05, 100);
  camera.position.set(0, 1.35, 4.7);
  camera.lookAt(0, 0, 0);

  scene.add(new THREE.HemisphereLight(0xdfe6ff, 0x11131c, 1.15));
  var key = new THREE.DirectionalLight(0xffffff, 2.6);
  key.position.set(3, 4, 5);
  scene.add(key);
  var fill = new THREE.DirectionalLight(0xffffff, 0.7);
  fill.position.set(-3, 1.5, 4);
  scene.add(fill);
  var rim = new THREE.DirectionalLight(0x8fa0ff, 1.6);
  rim.position.set(-3, -2, -4);
  scene.add(rim);

  var group = new THREE.Group();
  scene.add(group);

  var state = {
    idle: Math.random() * 10,
    scrollTarget: 0, scrollCurrent: 0,
    hoverTX: 0, hoverTY: 0, hoverCX: 0, hoverCY: 0,
    active: false, loaded: false, failed: false, raf: null
  };

  function resize() {
    var w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function tick() {
    state.raf = requestAnimationFrame(tick);
    state.idle += 0.0035;
    state.scrollCurrent += (state.scrollTarget - state.scrollCurrent) * 0.08;
    state.hoverCX += (state.hoverTX - state.hoverCX) * 0.08;
    state.hoverCY += (state.hoverTY - state.hoverCY) * 0.08;
    group.rotation.y = state.idle + state.scrollCurrent + state.hoverCX;
    group.rotation.x = state.hoverCY;
    renderer.render(scene, camera);
  }

  function start() {
    if (state.active) return;
    state.active = true;
    resize();
    tick();
  }
  function stop() {
    state.active = false;
    if (state.raf) cancelAnimationFrame(state.raf);
    state.raf = null;
  }

  function ensureLoaded() {
    if (state.loaded || state.failed) return;
    state.loaded = true;
    if (loaderEl) loaderEl.classList.add('visible');
    loader.load(
      modelUrl,
      function (gltf) {
        var model = gltf.scene;
        var box = new THREE.Box3().setFromObject(model);
        var size = new THREE.Vector3(); box.getSize(size);
        var center = new THREE.Vector3(); box.getCenter(center);
        var maxDim = Math.max(size.x, size.y, size.z) || 1;
        var scale = 2.15 / maxDim;
        model.scale.setScalar(scale);
        model.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
        group.add(model);
        if (loaderEl) loaderEl.classList.remove('visible');
        if (icon) icon.style.opacity = '0';
        canvas.classList.add('ready');
      },
      undefined,
      function (err) {
        state.failed = true;
        if (loaderEl) loaderEl.classList.remove('visible');
        stop();
        console.warn('D&A Gadgets: could not load 3D model', modelUrl, err);
      }
    );
  }

  return { pcard: pcard, state: state, resize: resize, start: start, stop: stop, ensureLoaded: ensureLoaded };
}

function init() {
  var pcards = document.querySelectorAll('.pcard[data-model]');
  var viewers = [];
  pcards.forEach(function (pcard) {
    var v = createViewer(pcard);
    if (v) viewers.push(v);
  });
  if (!viewers.length) return;

  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var v = viewers.filter(function (x) { return x.pcard === entry.target; })[0];
        if (!v) return;
        if (entry.isIntersecting) { v.ensureLoaded(); v.start(); }
        else { v.stop(); }
      });
    }, { rootMargin: '600px 0px 600px 0px', threshold: 0 });
    viewers.forEach(function (v) { io.observe(v.pcard); });
  } else {
    viewers.forEach(function (v) { v.ensureLoaded(); v.start(); });
  }

  viewers.forEach(function (v) {
    var visual = v.pcard.closest('.p-visual');
    if (!visual) return;
    visual.addEventListener('mousemove', function (e) {
      var r = visual.getBoundingClientRect();
      var px = (e.clientX - r.left) / r.width - 0.5;
      var py = (e.clientY - r.top) / r.height - 0.5;
      v.state.hoverTX = px * 0.8;
      v.state.hoverTY = clamp(-py * 0.5, -0.4, 0.4);
    });
    visual.addEventListener('mouseleave', function () {
      v.state.hoverTX = 0;
      v.state.hoverTY = 0;
    });
  });

  function updateScrollRotation() {
    viewers.forEach(function (v) {
      var block = v.pcard.closest('.product-block');
      if (!block) return;
      var r = block.getBoundingClientRect();
      var vh = window.innerHeight;
      var center = r.top + r.height / 2;
      var progress = clamp((vh / 2 - center) / (vh / 2 + r.height / 2), -1, 1);
      v.state.scrollTarget = progress * 0.6;
    });
  }
  var ticking = false;
  document.addEventListener('scroll', function () {
    if (!ticking) {
      ticking = true;
      requestAnimationFrame(function () { updateScrollRotation(); ticking = false; });
    }
  }, { passive: true });
  updateScrollRotation();

  window.addEventListener('resize', function () {
    viewers.forEach(function (v) { v.resize(); });
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
