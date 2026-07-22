import type { ReactNode } from 'react';

export interface CalloutProps {
  type?: 'tip' | 'note' | 'warning' | 'info';
  title?: string;
  children?: ReactNode;
}

const styles: Record<string, { bg: string; border: string; title: string; icon: string }> = {
  tip: {
    bg: 'bg-emerald-50',
    border: 'border-emerald-200',
    title: 'text-emerald-900',
    icon: '💡',
  },
  note: {
    bg: 'bg-cream-card',
    border: 'border-border',
    title: 'text-ink',
    icon: '📝',
  },
  warning: {
    bg: 'bg-amber-50',
    border: 'border-amber-200',
    title: 'text-amber-900',
    icon: '⚠️',
  },
  info: {
    bg: 'bg-red-50',
    border: 'border-red-200',
    title: 'text-accent',
    icon: 'ℹ️',
  },
};

const defaultTitles: Record<string, string> = {
  tip: 'Tip',
  note: 'Note',
  warning: 'Warning',
  info: 'Info',
};

export default function Callout({ type = 'note', title, children }: CalloutProps) {
  const s = styles[type] ?? styles.note;
  const label = title ?? defaultTitles[type];
  return (
    <div
      className={`rounded-lg border not-prose p-4 my-4 ${s.bg} ${s.border}`}
      role="note"
      aria-label={label}
    >
      <p className={`font-semibold mb-2 flex items-center gap-2 ${s.title}`}>
        <span aria-hidden>{s.icon}</span>
        {label}
      </p>
      <div className="text-ink-muted text-sm [&>*:last-child]:mb-0">{children}</div>
    </div>
  );
}
