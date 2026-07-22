import { useState, type ReactNode } from 'react';

export interface CodeBlockProps {
  lang?: string;
  title?: string;
  children?: ReactNode;
  metastring?: string;
  className?: string;
}

export default function CodeBlock({
  lang = 'text',
  title,
  children,
  className = '',
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const code = typeof children === 'string' ? children : (children as React.ReactElement)?.props?.children ?? '';

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignore
    }
  }

  return (
    <div className="my-4 rounded-lg overflow-hidden border border-slate-200 bg-slate-900">
      <div className="flex items-center justify-between px-3 py-2 bg-slate-800 border-b border-slate-700">
        {title ? (
          <span className="text-xs font-medium text-slate-400">{title}</span>
        ) : (
          <span className="text-xs font-medium text-slate-500 uppercase">{lang}</span>
        )}
        <button
          type="button"
          onClick={handleCopy}
          className="text-xs text-slate-400 hover:text-slate-200 transition-colors"
          aria-label={copied ? 'Copied' : 'Copy code'}
        >
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <pre className={`p-4 overflow-x-auto text-sm text-slate-100 ${className}`}>
        <code className={`language-${lang}`}>{children}</code>
      </pre>
    </div>
  );
}
