import { useEffect, useState } from 'react';
import { NavLink, Navigate, Route, Routes, Link } from 'react-router-dom';
import Overview from './pages/Overview';
import Timeline from './pages/Timeline';
import Incidents from './pages/Incidents';
import IncidentDetail from './pages/IncidentDetail';
import Evaluation from './pages/Evaluation';

const NAV = [
  { to: '/', label: 'Overview', end: true },
  { to: '/timeline', label: 'Timeline' },
  { to: '/incidents', label: 'Incidents' },
  { to: '/evaluation', label: 'Evaluation' },
];

function useTheme() {
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') ?? 'system');

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'system') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  return [theme, setTheme];
}

export default function App() {
  const [theme, setTheme] = useTheme();
  const next = { system: 'light', light: 'dark', dark: 'system' };
  const icon = { system: '◐', light: '☀', dark: '☾' };

  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand">
          <strong>agentic-copilot</strong>
          <span>anomaly detection &amp; investigation</span>
        </Link>
        <nav className="nav">
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end}>{item.label}</NavLink>
          ))}
        </nav>
        <div className="topbar-end">
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme(next[theme])}
            aria-label={`Theme: ${theme}. Switch to ${next[theme]}.`}
          >
            <span aria-hidden="true">{icon[theme]}</span> {theme}
          </button>
        </div>
      </header>

      <main className="page">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/timeline" element={<Timeline />} />
          <Route path="/incidents" element={<Incidents />} />
          <Route path="/incidents/:incidentId" element={<IncidentDetail />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
