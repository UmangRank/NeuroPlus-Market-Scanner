import sys
import requests
import pandas as pd
import numpy as np
import pyqtgraph as pg
from hurst import compute_Hc
from statsmodels.tsa.stattools import adfuller, kpss
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QTextEdit, QRadioButton, QButtonGroup, QGroupBox, QAbstractScrollArea, QHeaderView,
    QProgressBar, QDialog, QStackedWidget, QFileDialog, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QFont
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import os
import pygame
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtCore import QUrl
import csv

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class WindowControls(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Minimize button
        self.minimize_button = QPushButton("−")
        self.minimize_button.setFixedSize(30, 30)
        self.minimize_button.clicked.connect(self.minimize_window)
        self.minimize_button.setStyleSheet("""
            QPushButton {
                background-color: #1A1A1A;
                border: none;
                color: #00FFCC;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #333333;
            }
        """)

        # Close button
        self.close_button = QPushButton("×")
        self.close_button.setFixedSize(30, 30)
        self.close_button.clicked.connect(QApplication.quit)
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #1A1A1A;
                border: none;
                color: #00FFCC;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF0000;
                color: white;
            }
        """)

        layout.addWidget(self.minimize_button)
        layout.addWidget(self.close_button)
        self.setLayout(layout)

    def minimize_window(self):
        window = self.window()
        window.setWindowState(window.windowState() | Qt.WindowState.WindowMinimized)

class LoadingScreen(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loading")
        self.showFullScreen()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add window controls at the top
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addStretch()
        window_controls = WindowControls(self)
        controls_layout.addWidget(window_controls)
        main_layout.addLayout(controls_layout)
        
        # Content layout
        content_layout = QVBoxLayout()
        content_layout.addStretch()

        # Decorative top border with animation
        self.top_border = QLabel()
        self.top_border.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top_border.setStyleSheet("""
            color: #00FFCC;
            font-size: 24px;
            font-family: 'Courier New';
        """)
        content_layout.addWidget(self.top_border)
        
        # Animated hex codes display
        self.hex_display = QLabel()
        self.hex_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hex_display.setStyleSheet("""
            color: #00FFCC;
            font-family: 'Courier New';
            font-size: 14px;
            margin: 10px;
        """)
        content_layout.addWidget(self.hex_display)
        
        # Title label with enhanced sci-fi decoration
        title_label = QLabel("《 NEUROPULSE ENGINE v1.0 》")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            color: #00FFCC;
            font-size: 32px;
            font-weight: bold;
            font-family: 'Orbitron';
            margin: 20px;
            padding: 15px;
            border: 2px solid #00FFCC;
            border-radius: 5px;
            background-color: rgba(0, 255, 204, 0.1);
            text-align: center;
        """)
        content_layout.addWidget(title_label)

        # Animated system status display
        self.status_display = QLabel()
        self.status_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_display.setStyleSheet("""
            color: #00FFCC;
            font-size: 16px;
            font-family: 'Courier New';
            margin: 15px;
        """)
        content_layout.addWidget(self.status_display)
        
        # Matrix-style random character display
        self.matrix_display = QLabel()
        self.matrix_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.matrix_display.setStyleSheet("""
            color: #00FFCC;
            font-family: 'Courier New';
            font-size: 12px;
            margin: 10px;
        """)
        content_layout.addWidget(self.matrix_display)
        
        # Loading label with enhanced animation
        self.loading_label = QLabel()
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet("""
            color: #00FFCC;
            font-size: 18px;
            font-family: 'Courier New';
            margin: 15px;
        """)
        content_layout.addWidget(self.loading_label)
        
        # Progress bar with enhanced cyberpunk style
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #00FFCC;
                border-radius: 5px;
                text-align: center;
                color: #00FFCC;
                background-color: #000000;
                height: 25px;
                margin: 20px;
            }
            QProgressBar::chunk {
                background-color: #00FFCC;
                width: 20px;
                margin: 0.5px;
            }
        """)
        content_layout.addWidget(self.progress_bar)

        # System diagnostics display
        self.diagnostics_display = QLabel()
        self.diagnostics_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diagnostics_display.setStyleSheet("""
            color: #00FFCC;
            font-family: 'Courier New';
            font-size: 14px;
            margin: 15px;
        """)
        content_layout.addWidget(self.diagnostics_display)

        # Decorative bottom border with animation
        self.bottom_border = QLabel()
        self.bottom_border.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bottom_border.setStyleSheet("""
            color: #00FFCC;
            font-size: 24px;
            font-family: 'Courier New';
        """)
        content_layout.addWidget(self.bottom_border)
        
        content_layout.addStretch()
        main_layout.addLayout(content_layout)
        
        self.setLayout(main_layout)
        self.setStyleSheet("""
            QDialog {
                background-color: #000000;
                background-image: radial-gradient(circle at center, #001a14 0%, #000000 100%);
            }
        """)

        # Initialize animation timers
        self.border_timer = QTimer(self)
        self.border_timer.timeout.connect(self.update_borders)
        self.border_timer.start(500)

        self.hex_timer = QTimer(self)
        self.hex_timer.timeout.connect(self.update_hex_display)
        self.hex_timer.start(200)

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status_display)
        self.status_timer.start(1000)

        self.matrix_timer = QTimer(self)
        self.matrix_timer.timeout.connect(self.update_matrix_display)
        self.matrix_timer.start(100)

        self.loading_timer = QTimer(self)
        self.loading_timer.timeout.connect(self.update_loading_animation)
        self.loading_timer.start(300)

        self.diagnostics_timer = QTimer(self)
        self.diagnostics_timer.timeout.connect(self.update_diagnostics)
        self.diagnostics_timer.start(800)

        # Initialize animation states
        self.border_state = 0
        self.loading_state = 0
        self.matrix_chars = "ABCDEF0123456789"
        self.status_state = 0
        
    def update_borders(self):
        borders = [
            "╔══════════════════*****══════════════════╗",
            "╔══════════════════≡≡≡≡≡══════════════════╗",
            "╔══════════════════░░░░░══════════════════╗",
            "╔══════════════════▓▓▓▓▓══════════════════╗"
        ]
        bottom_borders = [
            "╚══════════════════*****══════════════════╝",
            "╚══════════════════≡≡≡≡≡══════════════════╝",
            "╚══════════════════░░░░░══════════════════╝",
            "╚══════════════════▓▓▓▓▓══════════════════╝"
        ]
        self.border_state = (self.border_state + 1) % len(borders)
        self.top_border.setText(borders[self.border_state])
        self.bottom_border.setText(bottom_borders[self.border_state])

    def update_hex_display(self):
        hex_values = [f"0x{format(np.random.randint(0, 16**6), '08x').upper()}" for _ in range(3)]
        self.hex_display.setText("\n".join(hex_values))

    def update_status_display(self):
        statuses = [
            "[ SYSTEM STATUS: INITIALIZING CORE MODULES ]",
            "[ NEURAL NETWORK: CALIBRATING ]",
            "[ QUANTUM PROCESSOR: ONLINE ]",
            "[ SECURITY PROTOCOLS: ACTIVE ]",
            "[ MARKET FEED: SYNCHRONIZING ]"
        ]
        self.status_state = (self.status_state + 1) % len(statuses)
        self.status_display.setText(statuses[self.status_state])

    def update_matrix_display(self):
        matrix_lines = []
        for _ in range(3):
            line = "".join(np.random.choice(list(self.matrix_chars)) for _ in range(40))
            matrix_lines.append(line)
        self.matrix_display.setText("\n".join(matrix_lines))

    def update_loading_animation(self):
        animations = [
            "▶ Loading System Components    ",
            "▶ Loading System Components •  ",
            "▶ Loading System Components •• ",
            "▶ Loading System Components •••"
        ]
        self.loading_state = (self.loading_state + 1) % len(animations)
        self.loading_label.setText(animations[self.loading_state])

    def update_diagnostics(self):
        diagnostics = [
            "[ MEMORY OPTIMIZATION: 97% ]",
            "[ NEURAL PATHWAYS: STABLE ]",
            "[ MARKET FEED: OPTIMAL ]",
            "[ CORE TEMPERATURE: NOMINAL ]"
        ]
        self.diagnostics_display.setText("\n".join(
            np.random.choice(diagnostics, 2, replace=False)
        ))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            QApplication.quit()

class SoundManager:
    def __init__(self):
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Path to the Sounds folder
        sounds_dir = os.path.join(script_dir, 'Sounds')
        
        # Initialize sound effects with error handling
        self.app_run = self._create_sound(os.path.join(sounds_dir, 'app_run.wav'))
        self.app_launch = self._create_sound(os.path.join(sounds_dir, 'app_launch.wav'))
        self.process_started = self._create_sound(os.path.join(sounds_dir, 'process_started.wav'))
        self.process_finished = self._create_sound(os.path.join(sounds_dir, 'Proccess_finished.wav'))
        self.click = self._create_sound(os.path.join(sounds_dir, 'click.wav'))
        self.error = self._create_sound(os.path.join(sounds_dir, 'Error.wav'))  # Add error sound

    def _create_sound(self, path):
        try:
            if not os.path.exists(path):
                print(f"Sound file not found: {path}")
                return None
            return pygame.mixer.Sound(path)
        except Exception as e:
            print(f"Error loading sound {path}: {str(e)}")
            return None

    def play_app_run(self):
        if self.app_run:
            self.app_run.play()

    def play_app_launch(self):
        if self.app_launch:
            self.app_launch.play()

    def play_process_started(self):
        if self.process_started:
            self.process_started.play()

    def play_process_finished(self):
        if self.process_finished:
            self.process_finished.play()

    def play_click(self):
        if self.click:
            self.click.play()

    def play_error(self):
        if self.error:
            self.error.play()

class ApplicationManager(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize sound manager
        self.sound_manager = SoundManager()
        # Play app run sound
        self.sound_manager.play_app_run()
        
        # Make window fullscreen
        self.showFullScreen()
        
        # Create stacked widget to manage loading screen and main window
        self.stacked_widget = QStackedWidget()
        
        # Create loading screen
        self.loading_screen = LoadingScreen()
        self.main_window = MexcTradingPairAnalyzer()
        
        # Add widgets to stacked widget
        self.stacked_widget.addWidget(self.loading_screen)
        self.stacked_widget.addWidget(self.main_window)
        
        # Set stacked widget as central widget
        self.setCentralWidget(self.stacked_widget)
        
        # Show loading screen first
        self.stacked_widget.setCurrentWidget(self.loading_screen)
        
        # Setup and start loading timer
        self.loading_timer = QTimer()
        self.loading_timer.timeout.connect(self.update_loading)
        self.load_progress = 0
        self.loading_timer.start(50)  # 50ms interval for smooth progression

    def update_loading(self):
        self.load_progress += 2
        self.loading_screen.progress_bar.setValue(self.load_progress)
        
        # Update loading label with dynamic text
        if self.load_progress < 33:
            self.loading_screen.loading_label.setText("Initializing modules...")
        elif self.load_progress < 66:
            self.loading_screen.loading_label.setText("Loading trading pairs...")
        else:
            self.loading_screen.loading_label.setText("Setting up interface...")
        
        # Switch to main window when loading completes
        if self.load_progress >= 100:
            self.loading_timer.stop()
            self.stacked_widget.setCurrentWidget(self.main_window)
            # Play app launch sound
            self.sound_manager.play_app_launch()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            QApplication.quit()

class DataFetchThread(QThread):
    progress_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()

    def __init__(self, trading_pairs, hurst_threshold, min_volume, timeframe, is_maximum=True):
        super().__init__()
        self.trading_pairs = trading_pairs
        self.hurst_threshold = hurst_threshold
        self.min_volume = min_volume
        self.timeframe = timeframe
        self.is_maximum = is_maximum
        self.is_running = True
        self.all_data = {}
        self.btc_data = None  # Store BTC data for correlation
        
        # Add batch processing
        self.batch_size = 50  # Adjust based on testing
        
        # Add caching
        self.cache = {}
        self.cache_duration = 3600  # 1 hour in seconds

    async def fetch_btc_data(self, session):
        """Fetch BTC data first for correlation comparison"""
        try:
            self.progress_signal.emit("Fetching BTC data for correlation...")
            url = "https://api.mexc.com/api/v3/klines"
            params = {
                'symbol': 'BTCUSDT',
                'interval': self.timeframe,
                'limit': 96 * 6
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        # Debug log
                        self.progress_signal.emit(f"Raw BTC data length: {len(data)}")
                        
                        df = pd.DataFrame(data, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 
                            'volume', 'close_time', 'ignore'
                        ])
                        df['close'] = df['close'].astype(float)
                        self.btc_data = df['close'].values
                        
                        # Debug logs
                        self.progress_signal.emit(f"BTC data sample: {self.btc_data[:5]}")
                        self.progress_signal.emit(f"BTC data shape: {self.btc_data.shape}")
                        return True
            
            self.progress_signal.emit("Failed to fetch BTC data")
            return False
        except Exception as e:
            self.progress_signal.emit(f"Error fetching BTC data: {str(e)}")
            return False

    def calculate_correlation(self, pair_closes):
        """Calculate correlation with BTC"""
        try:
            if self.btc_data is None:
                self.progress_signal.emit("No BTC data available for correlation")
                return 0.0
            
            # Convert to numpy arrays and ensure they're floats
            btc_prices = self.btc_data.astype(float)
            pair_prices = np.array(pair_closes).astype(float)
            
            # Ensure both arrays are the same length
            min_length = min(len(btc_prices), len(pair_prices))
            btc_prices = btc_prices[-min_length:]
            pair_prices = pair_prices[-min_length:]
            
            # Calculate percentage changes over multiple timeframes
            correlations = []
            for period in [1, 3, 6, 12, 24]:  # Multiple periods for more robust correlation
                try:
                    # Calculate rolling returns for both assets
                    btc_returns = np.diff(btc_prices[::period]) / btc_prices[:-period:period]
                    pair_returns = np.diff(pair_prices[::period]) / pair_prices[:-period:period]
                    
                    # Remove any infinite or NaN values
                    mask = np.isfinite(btc_returns) & np.isfinite(pair_returns)
                    btc_returns = btc_returns[mask]
                    pair_returns = pair_returns[mask]
                    
                    # Only calculate correlation if we have enough valid data points
                    if len(btc_returns) > 1 and len(pair_returns) > 1:
                        # Check for zero standard deviation
                        if np.std(btc_returns) > 0 and np.std(pair_returns) > 0:
                            corr = np.corrcoef(btc_returns, pair_returns)[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr)
                except Exception as e:
                    self.progress_signal.emit(f"Error in period {period}: {str(e)}")
                    continue
            
            if not correlations:
                self.progress_signal.emit(f"No valid correlations found for this pair")
                return 0.0
                
            # Take the average correlation across different timeframes
            avg_correlation = np.mean(correlations)
            correlation_value = float(avg_correlation * 100)  # Changed to float instead of int
            
            self.progress_signal.emit(f"Multi-timeframe correlation: {correlation_value:.2f} (from {len(correlations)} periods)")
            return correlation_value
            
        except Exception as e:
            self.progress_signal.emit(f"Error calculating correlation: {str(e)}")
            return 0.0

    async def fetch_all_pairs(self):
        """Fetch all pairs data concurrently"""
        try:
            self.progress_signal.emit("Starting batch data fetch...")
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=50)) as session:
                # First fetch BTC data
                if not await self.fetch_btc_data(session):
                    self.progress_signal.emit("Failed to fetch BTC data, continuing without correlation...")
                
                chunk_size = 100
                results = {}
                
                # Add semaphore for concurrent requests
                semaphore = asyncio.Semaphore(20)
                
                async def fetch_with_semaphore(pair):
                    if not self.is_running:  # Check if stopped
                        return pair, None
                    async with semaphore:
                        return await self.fetch_single_pair(session, pair)
                
                for i in range(0, len(self.trading_pairs), chunk_size):
                    if not self.is_running:  # Check if stopped
                        self.progress_signal.emit("Scan stopped by user.")
                        break
                        
                    chunk = self.trading_pairs[i:i + chunk_size]
                    self.progress_signal.emit(f"Fetching chunk {i//chunk_size + 1}/{len(self.trading_pairs)//chunk_size + 1}")
                    
                    # Create tasks for current chunk
                    tasks = [fetch_with_semaphore(pair) for pair in chunk]
                    
                    # Wait for current chunk to complete
                    chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    if not self.is_running:  # Check if stopped
                        break
                    
                    # Process chunk results immediately
                    chunk_data = {}
                    for result in chunk_results:
                        if isinstance(result, tuple):  # Valid result
                            pair, data = result
                            if data is not None:
                                chunk_data[pair] = data
                    
                    # Process the chunk data
                    for pair, data in chunk_data.items():
                        if not self.is_running:  # Check if stopped
                            break
                        try:
                            result = self.process_pair_data(pair, data)
                            if result:
                                self.result_signal.emit(result)
                        except Exception as e:
                            self.progress_signal.emit(f"Error processing {pair}: {str(e)}")
                    
                    # Short delay between chunks
                    if self.is_running:
                        await asyncio.sleep(0.5)
                
                if self.is_running:
                    self.progress_signal.emit(f"Data fetch and processing completed.")
                return results
        except Exception as e:
            self.progress_signal.emit(f"Error in fetch_all_pairs: {str(e)}")
            return {}

    async def fetch_single_pair(self, session, pair):
        """Async function to fetch data for a single pair"""
        url = "https://api.mexc.com/api/v3/klines"
        params = {
            'symbol': pair,
            'interval': self.timeframe,
            'limit': 96 * 6
        }
        
        retries = 3
        for attempt in range(retries):
            try:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 429:  # Rate limit
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    if response.status == 200:
                        data = await response.json()
                        if data and len(data) > 0:
                            # Convert data to proper format
                            df = pd.DataFrame(data)
                            df[4] = df[4].astype(float)  # Convert close prices to float
                            return pair, data
                    return pair, None
                
            except Exception as e:
                if attempt == retries - 1:
                    self.progress_signal.emit(f"Final error fetching {pair}: {str(e)}")
                    return pair, None
                await asyncio.sleep(1)
        return pair, None

    def process_pair_data(self, pair, raw_data):
        """Process the data for a single pair"""
        try:
            # Convert raw data to DataFrame with proper column names
            df = pd.DataFrame(raw_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 
                'volume', 'close_time', 'ignore'
            ])
            df['close'] = df['close'].astype(float)
            close_prices = df['close'].values
            
            # Calculate correlation with BTC
            correlation = self.calculate_correlation(close_prices)
            self.progress_signal.emit(f"Correlation for {pair}: {correlation:.2f}")
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['volume'] = df['volume'].astype(float)
            
            total_volume_in_usdt = (df['volume'] * df['close']).sum()
            
            if total_volume_in_usdt < self.min_volume:
                return None
                
            H_result = compute_Hc(df['close'].values, kind='price', simplified=True)
            hurst_exponent = H_result[0]
            
            # Modified threshold check based on type
            if self.is_maximum:
                if hurst_exponent >= self.hurst_threshold:
                    return None
            else:
                if hurst_exponent <= self.hurst_threshold:
                    return None
                
            adf_result = adfuller(df['close'].values)
            adf_pvalue = adf_result[1]
            
            # Update KPSS test implementation
            try:
                import warnings
                # Temporarily suppress the interpolation warning
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'The test statistic is outside of the range')
                    # Use price levels for KPSS test
                    kpss_result = kpss(df['close'].values, regression='c', nlags="auto")
                    kpss_pvalue = kpss_result[1]
                    
                    # Log the raw values for debugging
                    self.progress_signal.emit(f"KPSS for {pair}: stat={kpss_result[0]:.4f}, p-value={kpss_pvalue:.4f}")
                    
                    # Handle extreme p-values
                    if kpss_pvalue < 0.01:
                        kpss_pvalue = 0.01  # Cap at 0.01 for very significant results
                    elif kpss_pvalue > 0.99:
                        kpss_pvalue = 0.99  # Cap at 0.99 for very insignificant results
                
            except Exception as e:
                self.progress_signal.emit(f"KPSS test failed for {pair}: {str(e)}")
                kpss_pvalue = None  # Use None instead of 1.0 for failed tests
            
            # If KPSS test failed, skip this pair
            if kpss_pvalue is None:
                return None
            
            return {
                'pair': pair,
                'hurst_exponent': hurst_exponent,
                'total_volume': total_volume_in_usdt,
                'adf_pvalue': adf_pvalue,
                'kpss_pvalue': kpss_pvalue,
                'correlation': correlation,
                'data': df[['timestamp', 'close', 'volume', 'high', 'low']]
            }
            
        except Exception as e:
            self.progress_signal.emit(f"Error processing {pair}: {str(e)}")
            return None

    def run(self):
        """Main thread execution"""
        try:
            self.is_running = True
            self.progress_signal.emit("Starting data collection and analysis...")
            
            # Create and run the event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the fetch_all_pairs coroutine
            loop.run_until_complete(self.fetch_all_pairs())
            
            loop.close()
            
        except Exception as e:
            self.progress_signal.emit(f"Error in main loop: {str(e)}")
        finally:
            self.is_running = False
            self.finished_signal.emit()

    def stop(self):
        """Stop the processing"""
        self.progress_signal.emit("Stopping scan...")
        self.is_running = False

class MexcTradingPairAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MEXC Trading Pair Analyzer")
        
        # Initialize sound manager
        self.sound_manager = SoundManager()
        
        # Add timer for netrunner animation
        self.breach_timer = QTimer()
        self.breach_timer.timeout.connect(self.update_breach_animation)
        self.breach_sequence = 0
        self.breach_active = False

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add window controls at the top
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addStretch()
        window_controls = WindowControls(self)
        controls_layout.addWidget(window_controls)
        main_layout.addLayout(controls_layout)
        
        # Content layout
        content_layout = QHBoxLayout()
        
        # Add your existing layouts here
        left_section = QVBoxLayout()
        timeframe_group = QGroupBox("Timeframe")
        timeframe_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #00FFCC;
                border-radius: 5px;
                margin-top: 1ex;
                color: #00FFCC;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                color: #00FFCC;
            }
            QRadioButton {
                color: #00FFCC;
                spacing: 5px;
            }
            QRadioButton::indicator {
                width: 13px;
                height: 13px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #00FFCC;
                background: transparent;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #00FFCC;
                background: #00FFCC;
            }
        """)
        timeframe_layout = QHBoxLayout()
        self.timeframe_buttons = QButtonGroup()
        timeframes = ['1m', '5m', '15m', '30m']
        self.selected_timeframe = '15m'
        for tf in timeframes:
            radio_btn = QRadioButton(tf)
            self.timeframe_buttons.addButton(radio_btn)
            timeframe_layout.addWidget(radio_btn)
            radio_btn.clicked.connect(lambda _, t=tf: self.set_timeframe(t))
            if tf == '15m':
                radio_btn.setChecked(True)
        timeframe_group.setLayout(timeframe_layout)
        left_section.addWidget(timeframe_group)

        hurst_layout = QVBoxLayout()  # Change to vertical layout
        
        # Add radio button group
        hurst_type_group = QGroupBox("Hurst Threshold Type")
        hurst_type_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #00FFCC;
                border-radius: 5px;
                margin-top: 1ex;
                color: #00FFCC;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
            QRadioButton {
                color: #00FFCC;
                spacing: 5px;
            }
            QRadioButton::indicator {
                width: 13px;
                height: 13px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #00FFCC;
                background: transparent;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #00FFCC;
                background: #00FFCC;
            }
        """)
        
        radio_layout = QHBoxLayout()
        self.max_radio = QRadioButton("Maximum")
        self.min_radio = QRadioButton("Minimum")
        self.max_radio.setChecked(True)  # Default to maximum
        radio_layout.addWidget(self.max_radio)
        radio_layout.addWidget(self.min_radio)
        hurst_type_group.setLayout(radio_layout)
        hurst_layout.addWidget(hurst_type_group)
        
        # Add threshold input
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Hurst Threshold:"))
        self.hurst_input = QLineEdit("0.5")
        threshold_layout.addWidget(self.hurst_input)
        hurst_layout.addLayout(threshold_layout)
        
        left_section.addLayout(hurst_layout)

        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Min Volume (USDT):"))
        self.volume_input = QLineEdit("5000")
        volume_layout.addWidget(self.volume_input)
        left_section.addLayout(volume_layout)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Scan")
        self.start_btn.clicked.connect(self.start_scan)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Scan")
        self.stop_btn.clicked.connect(self.stop_scan)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        export_group = QGroupBox()
        export_layout = QHBoxLayout()
        
        # Create combo box for export type selection
        self.export_type_combo = QComboBox()
        self.export_type_combo.addItems(["Export All", "Export Favorites"])
        
        # Single export button
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.export_selected)
        self.export_btn.setEnabled(False)
        
        export_layout.addWidget(self.export_type_combo)
        export_layout.addWidget(self.export_btn)
        export_group.setLayout(export_layout)
        
        # Update the export group styling with a more visible dropdown arrow
        export_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #00FFCC;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 5px;
            }
            QPushButton {
                background-color: #1A1A1A;
                border: 2px solid #00FFCC;
                border-radius: 5px;
                color: #00FFCC;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #00FFCC;
                color: #000000;
            }
            QPushButton:disabled {
                background-color: #1A1A1A;
                border: 2px solid #00FFCC;
                color: #00FFCC;
                opacity: 0.5;
            }
            QComboBox {
                background-color: #1A1A1A;
                border: 2px solid #00FFCC;
                border-radius: 5px;
                color: #00FFCC;
                padding: 5px;
                padding-right: 20px;  /* Make room for arrow */
                min-width: 150px;
            }
            QComboBox:hover {
                border-color: #00FFCC;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
                background-color: transparent;
                border-left: 2px solid #00FFCC;
                margin-right: 5px;
            }
            QComboBox::down-arrow {
                width: 0;
                height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #00FFCC;
            }
            QComboBox:on {
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: #1A1A1A;
                border: 2px solid #00FFCC;
                color: #00FFCC;
                selection-background-color: #00FFCC;
                selection-color: #000000;
                outline: none;
            }
            QComboBox QListView {
                background-color: #1A1A1A;
                color: #00FFCC;
            }
            QComboBox QListView::item {
                background-color: #1A1A1A;
                color: #00FFCC;
                padding: 5px;
            }
            QComboBox QListView::item:hover {
                background-color: #00FFCC;
                color: #000000;
            }
            QComboBox QListView::item:selected {
                background-color: #00FFCC;
                color: #000000;
            }
        """)
        
        btn_layout.addWidget(export_group)

        left_section.addLayout(btn_layout)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: #000000;
                border: 2px solid #00FFCC;
                color: #00FFCC;
                font-family: 'Courier New';
                padding: 5px;
            }
            QScrollBar:vertical {
                border: none;
                background: #1A1A1A;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #00FFCC;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                background: none;
            }
        """)
        left_section.addWidget(QLabel("Log:"))
        left_section.addWidget(self.log_area)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "Favorite", "Pair", "Hurst Exponent", "Volume", 
            "ADF p-value", "KPSS p-value", "Correlation"
        ])
        self.results_table.itemSelectionChanged.connect(self.update_chart)
        self.results_table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)

        # Set fixed number of rows (starting with 20 rows)
        self.results_table.setRowCount(20)

        # Remove horizontal scrollbar and keep vertical scrollbar
        self.results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        # Expand the 4 columns
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # Favorite column
        header.setDefaultSectionSize(60)  # Set width for Favorite column
        for i in range(1, 7):  # Update range to include new column
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

        # Set fixed table height
        self.results_table.setFixedHeight(400)

        # Style the table and scrollbar
        self.results_table.setStyleSheet("""
            QTableWidget {
                background-color: #1A1A1A;
                border: 1px solid #00FFCC;
                color: #00FFCC;
                gridline-color: #00FFCC;
            }
            QTableWidget::item {
                border: 1px solid #00FFCC;
            }
            QTableWidget::indicator {
                width: 20px;
                height: 20px;
            }
            QTableWidget::indicator:unchecked {
                background-color: transparent;
                border: 2px solid #00FFCC;
                border-radius: 4px;
            }
            QTableWidget::indicator:checked {
                background-color: #00FFCC;
                border: 2px solid #00FFCC;
                border-radius: 4px;
            }
            QTableWidget QHeaderView::section {
                background-color: #1A1A1A;
                border: 1px solid #00FFCC;
                color: #00FFCC;
            }
            QScrollBar:vertical {
                border: none;
                background: #1A1A1A;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #00FFCC;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                background: none;
            }
            QTableCornerButton::section {
                background-color: #1A1A1A;
                border: 1px solid #00FFCC;
            }
        """)

        left_section.addWidget(QLabel("Qualified Pairs:"))
        left_section.addWidget(self.results_table)

        self.chart_widget = pg.PlotWidget()
        self.chart_widget.setBackground((10, 10, 10))

        content_layout.addLayout(left_section, 1)
        content_layout.addWidget(self.chart_widget, 2)
        
        main_layout.addLayout(content_layout)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.processed_pairs = {}

        # Sci-fi themed styles
        self.setStyleSheet("""
            QMainWindow {
                background-color: #000000;
                background-image: radial-gradient(circle at center, #001a14 0%, #000000 100%);
            }
            
            QLabel {
                color: #00FFCC;
                font-family: 'Orbitron', 'Courier New';
                text-transform: uppercase;
            }
            
            QPushButton {
                background-color: #1A1A1A;
                border: 2px solid #00FFCC;
                border-radius: 5px;
                color: #00FFCC;
                font-family: 'Orbitron', 'Courier New';
                font-size: 14px;
                padding: 5px;
                text-transform: uppercase;
            }
            
            QPushButton:hover {
                background-color: #00FFCC;
                color: #000000;
            }
            
            QTableWidget {
                background-color: #000000;
                border: 2px solid #00FFCC;
                gridline-color: #00FFCC;
                color: #00FFCC;
                font-family: 'Courier New';
            }
            
            QTableWidget::item:selected {
                background-color: #00FFCC;
                color: #000000;
            }
            
            QHeaderView::section {
                background-color: #1A1A1A;
                border: 1px solid #00FFCC;
                color: #00FFCC;
                font-family: 'Orbitron', 'Courier New';
            }
            
            QTextEdit {
                background-color: #000000;
                border: 2px solid #00FFCC;
                color: #00FFCC;
                font-family: 'Courier New';
                padding: 5px;
            }
            
            QLineEdit {
                background-color: #000000;
                border: 2px solid #00FFCC;
                border-radius: 5px;
                color: #00FFCC;
                font-family: 'Courier New';
                padding: 5px;
            }
        """)

        self.results_table.setStyleSheet("""
            QTableWidget {
                background-color: #1A1A1A;
                border: 1px solid #00FFCC;
                color: #00FFCC;
                gridline-color: #00FFCC;
            }
            QTableWidget::item {
                border: 1px solid #00FFCC;
            }
            QTableWidget::indicator {
                width: 20px;
                height: 20px;
            }
            QTableWidget::indicator:unchecked {
                background-color: transparent;
                border: 2px solid #00FFCC;
                border-radius: 4px;
            }
            QTableWidget::indicator:checked {
                background-color: #00FFCC;
                border: 2px solid #00FFCC;
                border-radius: 4px;
            }
            QTableWidget QHeaderView::section {
                background-color: #1A1A1A;
                border: 1px solid #00FFCC;
                color: #00FFCC;
            }
            QScrollBar:vertical {
                border: none;
                background: #1A1A1A;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #00FFCC;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                background: none;
            }
            QTableCornerButton::section {
                background-color: #1A1A1A;
                border: 1px solid #00FFCC;
            }
        """)

        # Modify input widgets to play click sound
        self.hurst_input.focusInEvent = lambda e: self.sound_manager.play_click()
        self.volume_input.focusInEvent = lambda e: self.sound_manager.play_click()
        
        # Add click sound to timeframe buttons
        for button in self.timeframe_buttons.buttons():
            button.clicked.connect(lambda: self.sound_manager.play_click())
        
        # Add click sound to table navigation
        self.results_table.keyPressEvent = self.table_key_press
        self.results_table.clicked.connect(lambda: self.sound_manager.play_click())

        # Initial "No Data" display
        self.setup_no_data_display()

    def setup_no_data_display(self):
        """Setup initial display when no data is present"""
        self.chart_widget.clear()
        layout = pg.GraphicsLayout()
        self.chart_widget.setCentralItem(layout)
        
        # Add empty row for spacing
        layout.addLabel('', row=0, col=0)
        
        # Add "SYSTEM AWAIT" at top
        self.await_label = layout.addLabel(
            '<div style="color: #00FFCC; font-family: Courier; font-size: 18pt; font-weight: bold; text-align: center;">'
            '╔═══════════════════════╗<br>'
            '║&nbsp;&nbsp;&nbsp;SYSTEM AWAIT&nbsp;&nbsp;&nbsp;║<br>'
            '╚═══════════════════════╝'
            '</div>', 
            row=1, col=0
        )
        
        # Add Militech and Netrunner-style messages with animation
        self.breach_label = layout.addLabel(
            '<div style="color: #00FFCC; font-family: Courier; font-size: 18pt; opacity: 0.7; text-align: center; margin-top: 100px;">'
            '[ NEUROPULSE ENGINE v1.0 ]<br><br>'
            '[ <span style="color: #00FFCC;">BREACH PROTOCOL INITIATED</span> ]<br><br>'
            '[ <span class="hex">1C 2D 55 7A BD E9</span> ]<br>'
            '[ <span class="buffer">BUFFER OVERFLOW: READY</span> ]<br><br>'
            '[ SYSTEM STATUS: <span style="color: #00FFCC;">ONLINE</span> ]<br><br>'
            '[ DAEMON STATUS: <span class="blink">ACTIVE</span> ]'
            '</div>',
            row=3, col=0
        )

        # Set background color
        self.chart_widget.setBackground((5, 5, 10))
        
        # Start animation
        self.breach_sequence = 0
        self.breach_active = True
        self.breach_timer.start(500)  # Update every 500ms

    def update_breach_animation(self):
        if not self.breach_active:
            return
        
        # Different hex codes to cycle through
        hex_sequences = [
            "1C 2D 55 7A BD E9",
            "FF 55 E9 2D 1C BD",
            "7A BD 1C 55 2D E9",
            "E9 1C BD 55 7A 2D"
        ]
        
        # Different buffer states
        buffer_states = [
            "BUFFER OVERFLOW: ACTIVE",
            "BUFFER OVERFLOW: ACTIVE",
            "BUFFER OVERFLOW: ACTIVE",
            "BUFFER OVERFLOW: ACTIVE"
        ]
        
        # Update the animation
        self.breach_sequence = (self.breach_sequence + 1) % 4
        
        # Create new label content with updated values
        content = (
            '<div style="color: #00FFCC; font-family: Courier; font-size: 18pt; opacity: 0.7; text-align: center; margin-top: 100px;">'
            '[ NEUROPULSE ENGINE v1.0 ]<br><br>'
            f'[ <span style="color: {["#00FFCC", "#00FFCC", "#00FFCC", "#00FFCC"][self.breach_sequence]};">BREACH PROTOCOL INITIATED</span> ]<br><br>'
            f'[ <span style="color: #00FFCC;">{hex_sequences[self.breach_sequence]}</span> ]<br>'
            f'[ <span style="color: #00FFCC;">{buffer_states[self.breach_sequence]}</span> ]<br><br>'
            '[ SYSTEM STATUS: <span style="color: #00FFCC;">ONLINE</span> ]<br><br>'
            f'[ DAEMON STATUS: <span style="color: {["#00FFCC", "#00FFCC", "#00FFCC", "#00FFCC"][self.breach_sequence]};">ACTIVE</span> ]'
            '</div>'
        )
        
        # Update the label
        self.breach_label.setText(content)

    def stop_breach_animation(self):
        self.breach_active = False
        self.breach_timer.stop()

    def set_timeframe(self, timeframe):
        self.selected_timeframe = timeframe

    def start_scan(self):
        # Play process started sound
        self.sound_manager.play_process_started()
        try:
            hurst_threshold = float(self.hurst_input.text())
            min_volume = float(self.volume_input.text())
            is_maximum = self.max_radio.isChecked()  # Check which threshold type is selected
        except ValueError:
            self.sound_manager.play_error()
            self.log_area.append("Please enter valid numbers!")
            return
        
        # Clear previous results and favorites
        self.results_table.clearContents()
        self.results_table.setRowCount(20)
        self.log_area.clear()
        self.processed_pairs.clear()
        self.chart_widget.clear()
        
        url = "https://api.mexc.com/api/v3/exchangeInfo"
        response = requests.get(url)
        trading_pairs = [symbol['symbol'] for symbol in response.json()['symbols']]
        
        # Pass the threshold type to DataFetchThread
        self.data_thread = DataFetchThread(trading_pairs, hurst_threshold, min_volume, self.selected_timeframe, is_maximum)
        self.data_thread.progress_signal.connect(self.log_area.append)
        self.data_thread.result_signal.connect(self.add_result_to_table)
        self.data_thread.finished_signal.connect(self.scan_completed)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.data_thread.start()

    def add_result_to_table(self, result):
        try:
            # Find first empty row or add new row
            row_position = 0
            for i in range(self.results_table.rowCount()):
                if self.results_table.item(i, 1) is None:  # Check second column (Pair)
                    row_position = i
                    break
                if i == self.results_table.rowCount() - 1:
                    row_position = i + 1
                    self.results_table.insertRow(row_position)

            # Create table items
            checkbox = QTableWidgetItem()
            checkbox.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            checkbox.setCheckState(Qt.CheckState.Unchecked)
            
            pair_item = QTableWidgetItem(result['pair'])
            hurst_item = QTableWidgetItem(f"{result['hurst_exponent']:.4f}")
            volume_item = QTableWidgetItem(f"{result['total_volume']:.2f}")
            adf_item = QTableWidgetItem(f"{result['adf_pvalue']:.4f}")
            kpss_item = QTableWidgetItem(f"{result['kpss_pvalue']:.4f}")  # Add KPSS item
            correlation_item = QTableWidgetItem(f"{result['correlation']:.2f}")
            
            # Set items in table
            self.results_table.setItem(row_position, 0, checkbox)
            self.results_table.setItem(row_position, 1, pair_item)
            self.results_table.setItem(row_position, 2, hurst_item)
            self.results_table.setItem(row_position, 3, volume_item)
            self.results_table.setItem(row_position, 4, adf_item)
            self.results_table.setItem(row_position, 5, kpss_item)  # Add KPSS column
            self.results_table.setItem(row_position, 6, correlation_item)
            
            # Store data for chart
            self.processed_pairs[result['pair']] = result['data']
            
            # Force table update
            self.results_table.viewport().update()

        except Exception as e:
            self.sound_manager.play_error()  # Add error sound
            self.log_area.append(f"Error adding to table: {str(e)}")

    def calculate_z_score(self, prices, window=20):
        """
        Calculate rolling z-score for the given prices.
        
        Args:
            prices (pd.Series): Price series
            window (int): Rolling window size for mean and standard deviation
        
        Returns:
            pd.Series: Z-score series
        """
        # Calculate rolling mean and standard deviation
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        # Calculate z-score: (price - rolling_mean) / rolling_std
        z_score = (prices - rolling_mean) / rolling_std
        
        return z_score

    def update_chart(self):
        if self.results_table.currentRow() < 0:
            self.setup_no_data_display()
            return
        self.stop_breach_animation()
        
        row = self.results_table.currentRow()
        if row < 0:  # No row selected
            return
        
        # Get pair from column 1 (second column) instead of column 0
        pair_item = self.results_table.item(row, 1)  # Changed from 0 to 1
        if not pair_item:
            return
            
        pair = pair_item.text()  # Get the text from the pair column
        if pair in self.processed_pairs:
            data = self.processed_pairs[pair]
            
            # Convert to numpy arrays
            timestamps = np.array([t.timestamp() for t in data['timestamp']])
            highs = np.array(data['high'].values, dtype=float)
            lows = np.array(data['low'].values, dtype=float)
            closes = np.array(data['close'].values, dtype=float)
            
            # Calculate z-score
            z_scores = self.calculate_z_score(pd.Series(closes))
            
            # Clear previous plots
            self.chart_widget.clear()
            
            # Create a layout for multiple plots with spacing
            layout = pg.GraphicsLayout()
            layout.setSpacing(20)
            self.chart_widget.setCentralItem(layout)
            
            # Add cyberpunk-style decorative elements using HTML formatting
            top_decor = layout.addLabel(
                '<div style="color: #00FFCC; font-family: Courier; font-size: 14pt; font-weight: bold; padding: 5px;">'
                '╔════ MARKET DATA ══════╗'
                '</div>'
            )
            
            # Price chart
            price_plot = layout.addPlot(row=1, col=0)
            price_plot.setLabel('left', '≡ PRICE ≡', color='#00FFCC')
            price_plot.getAxis('bottom').hide()
            price_plot.showGrid(x=True, y=True, alpha=0.3)
            
            # Enhanced border and background for price plot
            price_plot.getViewBox().setBackgroundColor(pg.mkColor(0, 10, 15))
            price_plot.getViewBox().setBorder(color='#00FFCC', width=2)
            
            # Cyberpunk-style title for price chart
            price_plot.setTitle(
                f"《 {pair} 》\n[ NEUROPULSE ANALYSIS ]", 
                color='#00FFCC', 
                size='14pt'
            )
            
            # Add glowing effect to grid
            price_plot.getAxis('left').setPen(pg.mkPen(color='#00FFCC', width=1.5))
            price_plot.getAxis('bottom').setPen(pg.mkPen(color='#00FFCC', width=1.5))
            
            # Create fill between high and low
            fill = pg.FillBetweenItem(
                curve1=pg.PlotDataItem(timestamps, highs),
                curve2=pg.PlotDataItem(timestamps, lows),
                brush=pg.mkBrush(color=(0, 255, 204, 30))
            )
            price_plot.addItem(fill)
            
            # Plot high and low prices as lines
            price_plot.plot(
                timestamps, 
                highs, 
                pen=pg.mkPen(color="#00FFCC", width=1),
                name='High'
            )
            
            price_plot.plot(
                timestamps, 
                lows, 
                pen=pg.mkPen(color="#00FFCC", width=1),
                name='Low'
            )
        
            
            # Z-Score plot with enhanced visuals
            z_score_plot = layout.addPlot(row=4, col=0)
            z_score_plot.setLabel('left', '≡ Z-SCORE ≡', color='#00FFCC')
            z_score_plot.showGrid(x=True, y=True, alpha=0.3)
            
            # Enhanced border and background for z-score plot
            z_score_plot.getViewBox().setBackgroundColor(pg.mkColor(0, 10, 15))
            z_score_plot.getViewBox().setBorder(color='#00FFCC', width=2)
            
            # Cyberpunk-style title for Z-Score
            z_score_plot.setTitle("《 STATISTICAL DEVIATION ANALYSIS 》", color='#00FFCC', size='12pt')
            
            # Add glowing effect to grid
            z_score_plot.getAxis('left').setPen(pg.mkPen(color='#00FFCC', width=1.5))
            z_score_plot.getAxis('bottom').setPen(pg.mkPen(color='#00FFCC', width=1.5))
            
            # Add reference lines for Z-Score levels
            levels = [-2, -1.5, -1, 1, 1.5, 2]
            colors = ['#00FFCC', '#00FFCC', '#00FFCC', '#00FFCC', '#00FFCC', '#00FFCC']
            for level, color in zip(levels, colors):
                z_score_plot.addLine(y=level, pen=pg.mkPen(color=color, style=Qt.PenStyle.DashLine, width=1))
            
            # Create zero line with more emphasis
            z_score_plot.addLine(y=0, pen=pg.mkPen(color='white', width=1.5))
            
            # Create fills for different Z-Score zones
            z_score_data = z_scores.values
            
            # Fill for extreme zones (|z| > 2)
            extreme_fill_pos = pg.FillBetweenItem(
                curve1=pg.PlotDataItem(timestamps, np.maximum(z_score_data, 2)),
                curve2=pg.PlotDataItem(timestamps, np.full_like(timestamps, 2)),
                brush=pg.mkBrush(color=(255, 51, 51, 50))
            )
            extreme_fill_neg = pg.FillBetweenItem(
                curve1=pg.PlotDataItem(timestamps, np.minimum(z_score_data, -2)),
                curve2=pg.PlotDataItem(timestamps, np.full_like(timestamps, -2)),
                brush=pg.mkBrush(color=(255, 51, 51, 50))
            )
            
            # Fill for moderate zones (1 < |z| < 2)
            moderate_fill_pos = pg.FillBetweenItem(
                curve1=pg.PlotDataItem(timestamps, np.clip(z_score_data, 1, 2)),
                curve2=pg.PlotDataItem(timestamps, np.full_like(timestamps, 1)),
                brush=pg.mkBrush(color=(255, 255, 102, 40))
            )
            moderate_fill_neg = pg.FillBetweenItem(
                curve1=pg.PlotDataItem(timestamps, np.clip(z_score_data, -2, -1)),
                curve2=pg.PlotDataItem(timestamps, np.full_like(timestamps, -1)),
                brush=pg.mkBrush(color=(255, 255, 102, 40))
            )
            
            # Add fills to plot
            z_score_plot.addItem(extreme_fill_pos)
            z_score_plot.addItem(extreme_fill_neg)
            z_score_plot.addItem(moderate_fill_pos)
            z_score_plot.addItem(moderate_fill_neg)
            
            # Plot Z-Score line on top
            z_score_plot.plot(
                timestamps, 
                z_score_data,
                pen=pg.mkPen(color="#00FFCC", width=2)
            )
            
            # Set Y axis range for Z-Score plot
            z_score_plot.setYRange(-3, 3)
            
            # Link x-axes of both plots
            z_score_plot.setXLink(price_plot)

            # Update log area with cyberpunk style
            self.log_area.clear()
            self.log_area.append("╔════ NEUROPULSE ══════════╗")
            self.log_area.append("║  MARKET ANALYSIS REPORT  ║")
            self.log_area.append("╠═══════════════════��══════╣")
            self.log_area.append(f"》 TRADING PAIR: {pair}")
            self.log_area.append(f"》 MEAN Z-SCORE: {z_scores.mean():.4f}")
            self.log_area.append(f"》 STD DEVIATION: {z_scores.std():.4f}")
            self.log_area.append(f"》 MAX Z-SCORE: {z_scores.max():.4f}")
            self.log_area.append(f"》 MIN Z-SCORE: {z_scores.min():.4f}")
            self.log_area.append("╚══════════════════════╝")

    def stop_scan(self):
        if hasattr(self, 'data_thread'):
            self.data_thread.stop()
            self.data_thread.wait()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(True)  # Enable the export button
        # Play process finished sound
        self.sound_manager.play_process_finished()

    def scan_completed(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(True)  # Only need to enable single export button now
        self.log_area.append("Scan completed.")
        # Play process finished sound
        self.sound_manager.play_process_finished()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            QApplication.quit()

    def table_key_press(self, event):
        if event.key() in [Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right]:
            self.sound_manager.play_click()
        super(QTableWidget, self.results_table).keyPressEvent(event)

    def export_selected(self):
        export_type = "all" if self.export_type_combo.currentText() == "Export All" else "favorites"
        self.export_to_csv(export_type=export_type)

    def export_to_csv(self, export_type="all"):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_name, _ = QFileDialog.getSaveFileName(self, "Save CSV File", current_directory, "CSV Files (*.csv);;All Files (*)")
        
        if file_name:
            try:
                with open(file_name, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    
                    # Write the header
                    writer.writerow(["Pair", "Hurst Exponent", "Volume", "ADF p-value", "KPSS p-value", "Correlation"])
                    
                    # Write the data rows
                    for row in range(self.results_table.rowCount()):
                        # Check if we should export this row
                        favorite_item = self.results_table.item(row, 0)
                        if favorite_item is None:
                            continue
                            
                        if export_type == "favorites" and favorite_item.checkState() != Qt.CheckState.Checked:
                            continue
                            
                        row_data = []
                        # Start from column 1 to skip the Favorite column
                        for column in range(1, self.results_table.columnCount()):
                            item = self.results_table.item(row, column)
                            row_data.append(item.text() if item else "")
                        writer.writerow(row_data)
                
                QMessageBox.information(self, "Success", "Data exported successfully!", 
                                      QMessageBox.StandardButton.Ok)
            except Exception as e:
                self.sound_manager.play_error()  # Add error sound
                QMessageBox.critical(self, "Error", f"Failed to export data: {e}", 
                                   QMessageBox.StandardButton.Ok)

# Global cache for pair tradability
tradability_cache = {}

def get_pair_details(symbol):
    if symbol in tradability_cache:
        return tradability_cache[symbol]  # Return cached result

    url = f'https://api.mexc.com/api/v3/exchangeInfo'
    
    # Sending request to get exchange info for all pairs
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Looping through all symbols and finding the requested symbol
        for symbol_info in data['symbols']:
            if symbol_info['symbol'] == symbol:
                # Check if the pair is tradable
                tradability_cache[symbol] = symbol_info['isSpotTradingAllowed']  # Cache the result
                return tradability_cache[symbol], symbol_info
        print(f"Symbol {symbol} not found.")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
    return None, None  # Return None if not found or error

class FetchPairDetailsWorker(QThread):
    result_signal = pyqtSignal(str, bool)  # Signal to send back the result

    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol

    def run(self):
        url = f'https://api.mexc.com/api/v3/exchangeInfo'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for symbol_info in data['symbols']:
                if symbol_info['symbol'] == self.symbol:
                    tradable = symbol_info['isSpotTradingAllowed']
                    self.result_signal.emit(self.symbol, tradable)  # Emit the result
                    return
        self.result_signal.emit(self.symbol, None)  # Emit None if not found or error

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ApplicationManager()
    window.setWindowTitle("MEXC Trading Pair Analyzer")
    window.show()
    sys.exit(app.exec())