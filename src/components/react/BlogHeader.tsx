/**
 * Distill-style academic metadata header for MDX blog posts.
 * Three-column grid: AUTHORS | AFFILIATIONS | PUBLISHED.
 */

export interface BlogAuthor {
  name: string;
  affiliation: string;
}

export interface BlogHeaderProps {
  title?: string;
  authors: BlogAuthor[];
  publishedDate: string;
}

export default function BlogHeader({
  title,
  authors,
  publishedDate,
}: BlogHeaderProps) {
  return (
    <header className="w-full border-y border-gray-200 py-6 my-8">
      {title ? (
        <h1 className="text-4xl font-bold tracking-tight text-gray-900 mb-6">
          {title}
        </h1>
      ) : null}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div>
          <div className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-3">
            Authors
          </div>
          <div className="flex flex-col gap-2">
            {authors.map((author, i) => (
              <div key={i} className="text-gray-900 font-medium leading-relaxed">
                {author.name}
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-3">
            Affiliations
          </div>
          <div className="flex flex-col gap-2">
            {authors.map((author, i) => (
              <div key={i} className="text-gray-700 leading-relaxed">
                {author.affiliation}
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-3">
            Published
          </div>
          <div className="text-gray-700 leading-relaxed">{publishedDate}</div>
        </div>
      </div>
    </header>
  );
}
