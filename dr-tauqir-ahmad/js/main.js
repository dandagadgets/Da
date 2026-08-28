if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  document.documentElement.classList.add('reduce-motion');
}

// Mobile nav toggle
const navToggle = document.getElementById('navToggle');
const navLinks = document.getElementById('navLinks');

if (navToggle && navLinks) {
  navToggle.addEventListener('click', () => {
    const isOpen = navLinks.classList.toggle('is-open');
    navToggle.setAttribute('aria-expanded', String(isOpen));
  });

  navLinks.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      navLinks.classList.remove('is-open');
      navToggle.setAttribute('aria-expanded', 'false');
    });
  });
}

// Animated count-up for stat numbers
function animateCount(el) {
  const target = parseFloat(el.dataset.count);
  if (Number.isNaN(target)) return;
  const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals, 10) : 0;
  const suffix = el.dataset.suffix || '';
  const prefix = el.dataset.prefix || '';
  const duration = 1200;

  if (document.documentElement.classList.contains('reduce-motion')) {
    el.textContent = prefix + target.toFixed(decimals) + suffix;
    return;
  }

  let startTime = null;
  function step(ts) {
    if (!startTime) startTime = ts;
    const progress = Math.min((ts - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = prefix + (target * eased).toFixed(decimals) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// Scroll reveal (also triggers count-up for any [data-count] inside)
const revealEls = document.querySelectorAll('[data-reveal]');

if ('IntersectionObserver' in window && revealEls.length) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          entry.target.querySelectorAll('[data-count]').forEach(animateCount);
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15, rootMargin: '0px 0px -60px 0px' }
  );
  revealEls.forEach((el) => observer.observe(el));
} else {
  revealEls.forEach((el) => {
    el.classList.add('is-visible');
    el.querySelectorAll('[data-count]').forEach(animateCount);
  });
}

// Floating "Book Appointment" pill — shown once past the hero, hidden again over the booking form
const floatCta = document.getElementById('floatCta');
const heroEl = document.getElementById('hero');
const bookEl = document.getElementById('book');

if (floatCta && heroEl && bookEl && 'IntersectionObserver' in window) {
  const floatCtaLink = floatCta.querySelector('a');
  let heroVisible = true;
  let bookVisible = false;
  const ctaObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.target === heroEl) heroVisible = entry.isIntersecting;
        if (entry.target === bookEl) bookVisible = entry.isIntersecting;
      });
      const shown = !heroVisible && !bookVisible;
      floatCta.classList.toggle('is-shown', shown);
      floatCta.setAttribute('aria-hidden', String(!shown));
      if (floatCtaLink) floatCtaLink.tabIndex = shown ? 0 : -1;
    },
    { threshold: 0.1 }
  );
  ctaObserver.observe(heroEl);
  ctaObserver.observe(bookEl);
}

// Appointment request form (client-side confirmation, no backend)
const bookForm = document.getElementById('bookForm');
const bookConfirm = document.getElementById('bookConfirm');

if (bookForm && bookConfirm) {
  bookForm.addEventListener('submit', (e) => {
    e.preventDefault();
    bookForm.hidden = true;
    bookConfirm.hidden = false;
    bookConfirm.focus?.();
  });
}
