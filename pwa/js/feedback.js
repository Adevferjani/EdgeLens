// Short messages (toasts), error reporting and the engine label in the top bar.

import { $, h } from './util.js';
import { logError } from './diagnostics.js';

/** action: {label, run} or a list of them; returns the toast node. */
export function toast(text, action, { sticky = false } = {}) {
  const box = $('#toasts');
  const actions = [action].flat().filter(Boolean);
  const node = h('div', { class: 'toast', role: 'status' }, h('span', {}, text),
    actions.length ? h('span', { class: 'toast-actions' }, actions.map((a) => h('button', {
      type: 'button',
      class: `btn small${a.quiet ? ' quiet' : ''}`,
      onclick: () => {
        node.remove();
        a.run();
      },
    }, a.label))) : null);
  box.append(node);
  if (!sticky) setTimeout(() => node.remove(), actions.length ? 7500 : 2500);
  return node;
}

export function showError(err) {
  console.error(err);
  if (!err?.logged) logError('ui', err?.message || err);
  toast(err?.message || String(err));
}

export const setEngineChip = (text) => {
  $('#engineChip').textContent = text.toUpperCase();
};
