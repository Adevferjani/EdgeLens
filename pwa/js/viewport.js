// Layout measurements that CSS cannot do alone: the iPhone home-screen height
// fix and the size of the bottom/side bar (used to place the toasts).

import { $ } from './util.js';
import { isAppleMobile } from './inference.js';

// iPhone home-screen apps draw under the translucent status bar, but iOS can
// report the screen as shorter by the status-bar height (100dvh), which left an
// empty strip at the bottom. Measure it and use the full screen height instead.
export let viewportInfo = null;
function fixStandaloneHeight() {
  const standalone = navigator.standalone === true || matchMedia('(display-mode: standalone)').matches;
  const probe = document.createElement('div');
  probe.style.cssText = 'position:fixed;top:0;left:0;width:0;height:100dvh;visibility:hidden;pointer-events:none';
  document.body.append(probe);
  const dvh = Math.round(probe.getBoundingClientRect().height);
  probe.remove();
  const portrait = matchMedia('(orientation: portrait)').matches;
  const full = portrait ? Math.max(screen.width, screen.height) : Math.min(screen.width, screen.height);
  const gap = full - dvh;
  const apply = standalone && isAppleMobile() && gap > 0 && gap <= 80;
  document.documentElement.style.setProperty('--app-h', apply ? `${full}px` : '');
  if (!apply) document.documentElement.style.removeProperty('--app-h');
  viewportInfo = { standalone, dvh, innerHeight: window.innerHeight, screen: full, corrected: apply ? gap : 0 };
}
fixStandaloneHeight();
window.addEventListener('orientationchange', () => setTimeout(fixStandaloneHeight, 300));

// Toasts sit just above the bottom bar, whose height depends on the mode; with
// the bar on the right (same query as in app.css) they are centred beside it.
const SIDE_BAR_QUERY = '(orientation: landscape) and (max-height: 540px), (orientation: landscape) and (hover: hover) and (pointer: fine)';
{
  const dock = $('.dock');
  const setDockHeight = () => {
    const portrait = !matchMedia(SIDE_BAR_QUERY).matches;
    document.documentElement.style.setProperty('--dock-h', `${portrait ? dock.offsetHeight : 0}px`);
    document.documentElement.style.setProperty('--dock-w', `${portrait ? 0 : dock.offsetWidth}px`);
  };
  new ResizeObserver(setDockHeight).observe(dock);
  setDockHeight();
}
