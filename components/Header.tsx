
import React from 'react';

const Header: React.FC = () => {
    return (
        <header className="hidden md:block bg-gray-900/80 backdrop-blur-sm sticky top-0 z-40">
            <div className="container mx-auto px-4 py-4">
                <div className="flex justify-between items-center">
                    <div className="flex items-center space-x-3">
                         <div className="w-8 h-8 bg-gradient-to-tr from-purple-500 to-indigo-600 rounded-lg"></div>
                        <span className="text-xl font-bold text-white">Arnold Jourdan</span>
                    </div>
                </div>
            </div>
        </header>
    );
};

export default Header;
