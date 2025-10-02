
import React, { useState, useMemo } from 'react';
import { TOOLS } from '../constants';
import { ToolCategory } from '../types';
import type { Tool, ToolCategoryValue } from '../types';
import FeatureCard from './FeatureCard';

interface DashboardProps {
    onSelectTool: (tool: Tool) => void;
}

const Dashboard: React.FC<DashboardProps> = ({ onSelectTool }) => {
    const [showWelcome, setShowWelcome] = useState(true);

    const groupedTools = TOOLS.reduce((acc, tool) => {
        if (!acc[tool.category]) {
            acc[tool.category] = [];
        }
        acc[tool.category].push(tool);
        return acc;
    }, {} as Record<ToolCategoryValue, Tool[]>);

    const categoryOrder = useMemo(() => Object.values(ToolCategory), []);

    return (
        <div className="space-y-12">
            {showWelcome && (
                <div className="bg-gray-800/60 border border-purple-500/30 p-6 rounded-lg relative">
                    <button 
                        onClick={() => setShowWelcome(false)}
                        className="absolute top-3 right-3 text-gray-500 hover:text-white transition-colors"
                        aria-label="Cerrar bienvenida"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                    <div className="flex items-start space-x-4">
                        <div className="flex-shrink-0 text-purple-400 mt-1">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-white">¡Bienvenido al AI Designer Toolkit!</h2>
                            <p className="mt-2 text-gray-300">
                                Esta es tu caja de herramientas de diseño inteligente. Explora las categorías a continuación para empezar. Puedes generar imágenes desde texto, crear paletas de colores, obtener críticas de tus diseños y mucho más.
                            </p>
                            <p className="mt-2 text-gray-300">
                                Simplemente haz clic en cualquier herramienta para comenzar a crear.
                            </p>
                        </div>
                    </div>
                </div>
            )}
            <div className="text-center">
                 <h1 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-indigo-500">
                    AI Designer Toolkit
                </h1>
                <p className="mt-4 text-lg text-gray-400 max-w-2xl mx-auto">
                    Herramientas inteligentes para potenciar la creatividad y optimizar el flujo de trabajo de Arnold Jourdan.
                </p>
            </div>
            
            {categoryOrder.map(category => (
                <div key={category}>
                    <h2 className="text-2xl font-semibold mb-6 border-l-4 border-purple-500 pl-4">{category}</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                        {groupedTools[category]?.map(tool => (
                            <FeatureCard key={tool.id} tool={tool} onClick={() => onSelectTool(tool)} />
                        ))}
                    </div>
                </div>
            ))}
        </div>
    );
};

export default Dashboard;
