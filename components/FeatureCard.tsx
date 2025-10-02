
import React from 'react';
import type { Tool } from '../types';

interface FeatureCardProps {
    tool: Tool;
    onClick: () => void;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ tool, onClick }) => {
    const cardClasses = tool.isImplemented 
        ? "bg-gray-800/50 cursor-pointer group hover:bg-gray-800 hover:border-purple-500/50"
        : "bg-gray-800/20 cursor-not-allowed opacity-60";

    return (
        <div 
            onClick={tool.isImplemented ? onClick : undefined}
            className={`rounded-lg p-6 flex flex-col transition-all duration-300 border border-transparent ${cardClasses}`}
        >
            <div className="flex items-center space-x-4 mb-4">
                <div className={`transition-colors ${tool.isImplemented ? 'text-purple-400 group-hover:text-purple-300' : 'text-gray-500'}`}>
                    {tool.icon}
                </div>
                <h3 className="font-semibold text-lg text-gray-100">{tool.title}</h3>
                 {tool.isImplemented ? 
                    <span className="text-xs bg-green-500/20 text-green-300 px-2 py-0.5 rounded-full">Disponible</span> :
                    <span className="text-xs bg-yellow-500/20 text-yellow-300 px-2 py-0.5 rounded-full">Próximamente</span>
                 }
            </div>
            <p className="text-gray-400 text-sm flex-grow">{tool.description}</p>
        </div>
    );
};

export default FeatureCard;