export default function Toasts({ toasts }) {
  if (!toasts.length) return null;
  return (
    <div className="toasts">
      {toasts.map(t => (
        <div key={t.id} className="toast">
          <div className={`toast-dot ${t.type}`} />
          <span>{t.msg}</span>
        </div>
      ))}
    </div>
  );
}
