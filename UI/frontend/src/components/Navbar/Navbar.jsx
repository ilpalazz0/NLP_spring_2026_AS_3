import { NavLink } from 'react-router-dom'
import './Navbar.css'

const NAV_ITEMS = [
  { to: '/analyzer',   icon: '', label: 'Analyzer'   },
  { to: '/dataset',    icon: '', label: 'Dataset'    },
  { to: '/evaluation', icon: '', label: 'Evaluation' },
  { to: '/word2vec',   icon: '', label: 'Word2Vec'   },
  { to: '/glove',      icon: '', label: 'GloVe'      },
]

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar__brand">
        <span className="navbar__brand-text">Literally the Most Accurate Sentiment Analayzer Ever 100% Verified</span>
      </div>

      <div className="navbar__divider" />

      <ul className="navbar__list">
        {NAV_ITEMS.map(item => (
          <li key={item.to}>
            <NavLink
              to={item.to}
              className={({ isActive }) =>
                `navbar__item ${isActive ? 'navbar__item--active' : ''}`
              }
            >
              <span className="navbar__item-icon">{item.icon}</span>
              <span className="navbar__item-label">{item.label}</span>
            </NavLink>
          </li>
        ))}
      </ul>

      <div className="navbar__footer">
      </div>
    </nav>
  )
}

export default Navbar
