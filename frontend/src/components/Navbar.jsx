import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Brain, Upload, Search, Lightbulb, Database } from 'lucide-react'

const Navbar = () => {
  const location = useLocation()

  const navItems = [
    { path: '/query', label: 'Query', icon: Search },
    { path: '/upload', label: 'Upload', icon: Upload },
    { path: '/recommendations', label: 'Recommendations', icon: Lightbulb },
  ]

  return (
    <nav className="bg-dark-card border-b border-dark-border sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold gradient-text">AI Research Copilot</span>
          </Link>

          {/* Navigation */}
          <div className="flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.path
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                    isActive
                      ? 'bg-primary-600/20 text-primary-400'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-dark-border'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="font-medium">{item.label}</span>
                </Link>
              )
            })}
          </div>

          {/* Status indicator */}
          <div className="flex items-center space-x-2 text-sm text-slate-500">
            <Database className="w-4 h-4" />
            <span>Endee Vector DB</span>
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
