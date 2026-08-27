(function () {
  gsap.registerPlugin(ScrollTrigger);

  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- Nav ---------- */
  var nav = document.getElementById('nav');
  var navToggle = document.getElementById('navToggle');
  var navLinks = document.getElementById('navLinks');
  var showcaseEl = document.getElementById('showcase');

  gsap.set(nav, { opacity: 1 });

  navToggle.addEventListener('click', function () {
    var open = navLinks.classList.toggle('is-open');
    navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  });
  navLinks.querySelectorAll('a').forEach(function (a) {
    a.addEventListener('click', function () {
      navLinks.classList.remove('is-open');
      navToggle.setAttribute('aria-expanded', 'false');
    });
  });

  ScrollTrigger.create({
    trigger: showcaseEl,
    start: 'top top',
    onEnter: function () { nav.classList.add('is-visible'); },
    onLeaveBack: function () { nav.classList.remove('is-visible'); }
  });
  ScrollTrigger.create({
    start: 100,
    end: 99999,
    onUpdate: function (self) {
      nav.classList.toggle('is-solid', self.scroll() > 120);
    }
  });

  /* ---------- Generic reveal-on-scroll ---------- */
  document.querySelectorAll('[data-reveal]').forEach(function (el) {
    ScrollTrigger.create({
      trigger: el,
      start: 'top 88%',
      once: true,
      onEnter: function () { el.classList.add('is-visible'); }
    });
  });

  /* ---------- Stat counters ---------- */
  function animateCounter(el) {
    var raw = el.dataset.count;
    var target = parseFloat(raw);
    var prefix = el.dataset.prefix || '';
    var suffix = el.dataset.suffix || '';
    var decimals = (raw.split('.')[1] || '').length;
    var obj = { val: 0 };
    gsap.to(obj, {
      val: target,
      duration: reduceMotion ? 0 : 1.8,
      ease: 'power2.out',
      onUpdate: function () {
        el.textContent = prefix + obj.val.toFixed(decimals) + suffix;
      }
    });
  }
  document.querySelectorAll('.stat__value').forEach(function (el) {
    ScrollTrigger.create({
      trigger: el,
      start: 'top 88%',
      once: true,
      onEnter: function () { animateCounter(el); }
    });
  });

  /* ---------- Market charts reveal ---------- */
  var marketSection = document.getElementById('market');
  if (marketSection) {
    ScrollTrigger.create({
      trigger: marketSection,
      start: 'top 65%',
      once: true,
      onEnter: function () {
        document.querySelectorAll('.barchart__bar').forEach(function (bar, i) {
          bar.style.transitionDelay = (i * 0.08) + 's';
          bar.classList.add('is-visible');
        });
        var stack = document.getElementById('stackBar');
        if (stack) stack.classList.add('is-visible');
        var eq = document.querySelector('.eqchart');
        if (eq) eq.classList.add('is-visible');
      }
    });
  }

  /* ---------- Idle floating bob for stage + craft/origin visuals ---------- */
  if (!reduceMotion) {
    var stageWrap = document.querySelector('.stage-wrap');
    if (stageWrap) {
      gsap.to(stageWrap, { y: 14, duration: 3.2, ease: 'sine.inOut', yoyo: true, repeat: -1 });
    }
    document.querySelectorAll('.float-box').forEach(function (box, i) {
      gsap.to(box, { y: 10, duration: 3 + (i % 3) * 0.4, ease: 'sine.inOut', yoyo: true, repeat: -1, delay: i * 0.2 });
    });
  }

  /* ---------- Pinned 3D showcase ---------- */
  var stage = document.getElementById('stage');
  var ringItems = gsap.utils.toArray('.ring-item');
  var captions = gsap.utils.toArray('.cap');
  var intro = document.getElementById('showcaseIntro');

  if (stage && ringItems.length) {
    ringItems.forEach(function (item) {
      gsap.set(item, { opacity: 0, x: -220, rotateY: -70, scale: 0.85 });
    });

    var tl = gsap.timeline({ defaults: { ease: 'power2.inOut' } });

    tl.to(intro, { opacity: 0, y: -50, duration: 1 }, 0)
      .to(intro, { display: 'none', duration: 0.01 }, 1);

    var unit = 4;
    ringItems.forEach(function (item, i) {
      var t = 0.6 + i * unit;
      var cap = captions[i];

      tl.to(item, { opacity: 1, x: 0, rotateY: 0, scale: 1, duration: 1.1 }, t)
        .to(item, { rotateY: 16, duration: unit - 2.2 }, t + 1.1)
        .to(item, { opacity: 0, x: 220, rotateY: 70, scale: 0.85, duration: 1.1 }, t + unit - 1.1);

      if (cap) {
        tl.fromTo(cap, { opacity: 0, y: 26 }, { opacity: 1, y: 0, duration: 0.7 }, t + 0.3)
          .to(cap, { opacity: 0, y: -18, duration: 0.7 }, t + unit - 1.3);
      }
    });

    ScrollTrigger.create({
      trigger: showcaseEl,
      start: 'top top',
      end: 'bottom bottom',
      pin: '.showcase__pin',
      scrub: 1,
      animation: tl
    });
  }

  ScrollTrigger.refresh();
})();
