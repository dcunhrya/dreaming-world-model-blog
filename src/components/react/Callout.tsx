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
    title: 'text-emerald-800',
    icon: '💡',
  },
  note: {
    bg: 'bg-slate-50',
    border: 'border-slate-200',
    title: 'text-slate-800',
    icon: '📝',
  },
  warning: {
    bg: 'bg-amber-50',
    border: 'border-amber-200',
    title: 'text-amber-800',
    icon: '⚠️',
  },
  info: {
    bg: 'bg-blue-50',
    border: 'border-blue-200',
    title: 'text-blue-800',
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
      className={`rounded-lg border ${s.bg} ${s.border} p-4 my-4`}
      role="note"
      aria-label={label}
    >
      <p className={`font-semibold ${s.title} mb-2 flex items-center gap-2`}>
        <span aria-hidden>{s.icon}</span>
        {label}
      </p>
      <div className="text-slate-700 prose prose-sm max-w-none [&>*:last-child]:mb-0">
        {children}
      </div>
    </div>
  );
}
