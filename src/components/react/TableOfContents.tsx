import { useEffect, useState } from 'react';

export interface TableOfContentsProps {
  sections: { id: string; title: string }[];
}

export default function TableOfContents({ sections }: TableOfContentsProps) {
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const id = entry.target.getAttribute('id');
            if (id) setActiveId(id);
            break;
          }
        }
      },
      {
        rootMargin: '-80px 0px -80% 0px',
        threshold: 0,
      }
    );

    const elements = sections.map(({ id }) => document.getElementById(id)).filter(Boolean);
    elements.forEach((el) => el && observer.observe(el));

    return () => observer.disconnect();
  }, [sections]);

  return (
    <nav aria-label="Table of contents" className="space-y-1">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
        On this page
      </h2>
      <ul className="space-y-1">
        {sections.map(({ id, title }) => (
          <li key={id}>
            <a
              href={`#${id}`}
              className={`block text-sm py-1.5 px-2 rounded-md transition-colors ${
                activeId === id
                  ? 'bg-slate-200 text-slate-900 font-medium'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              {title}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
