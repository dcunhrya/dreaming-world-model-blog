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
  const code =
    typeof children === 'string'
      ? children
      : (children as React.ReactElement)?.props?.children ?? '';

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
    <div className="not-prose my-4 rounded-lg border border-border overflow-hidden bg-cream-card shadow-sm">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-cream-muted/60">
        {title ? (
          <span className="text-xs font-mono text-ink-light">{title}</span>
        ) : (
          <span className="text-xs font-mono text-ink-light uppercase">{lang}</span>
        )}
        <button
          type="button"
          onClick={handleCopy}
          className="text-xs text-accent hover:text-accent-light transition-colors"
          aria-label={copied ? 'Copied' : 'Copy code'}
        >
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <pre className={`p-4 overflow-x-auto text-sm font-mono text-ink ${className}`}>
        <code className={`language-${lang}`}>{children}</code>
      </pre>
    </div>
  );
}
