from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit  # Added SocketIO
import os
import logging
import time
import threading
from datetime import datetime
import random


try:
    from yolov8_detector import detect_vehicles
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO detector not available, using fallback detection")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # Added SocketIO


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app.config.update({
    'UPLOAD_FOLDER': 'uploads',
    'MAX_CONTENT_LENGTH': 200 * 1024 * 1024,  # 200MB
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'gif', 'bmp'},
    'SECRET_KEY': 'traffic-management-system-2025'
})

# Create directories
for directory in ['uploads', 'templates', 'models']:
    os.makedirs(directory, exist_ok=True)


users_db = {}
sessions = {}
emergency_signals = []  



class TrafficLaneManager:
    """4-lane traffic signal management system"""
    
    def __init__(self):
        self.lanes = {
            1: {'name': 'North', 'vehicles': 0, 'status': 'red', 'priority': 0},
            2: {'name': 'East', 'vehicles': 0, 'status': 'red', 'priority': 0},
            3: {'name': 'South', 'vehicles': 0, 'status': 'red', 'priority': 0},
            4: {'name': 'West', 'vehicles': 0, 'status': 'red', 'priority': 0}
        }
        
        self.current_green_lane = None
        self.timer_active = False
        self.cycle_active = False
        self.emergency_active = False  
        self.signal_log = []
        
        
        self.min_green_time = 15
        self.max_green_time = 90
        self.yellow_time = 3
        self.all_red_time = 2
        self.base_time_per_vehicle = 4

    
    def activate_emergency_mode(self, lane_id=1):
        """Activate emergency mode for specified lane"""
        try:
            self.stop_cycle()  
            
            self.emergency_active = True
            self.current_green_lane = lane_id
            self.timer_active = True
            
            
            self._set_all_lanes_red()
            self.lanes[lane_id]['status'] = 'green'
            self.lanes[lane_id]['priority'] = 10  
            
            self._log_event('EMERGENCY_ACTIVATED', f"Emergency mode activated for Lane {lane_id}")
            
            
            emergency_thread = threading.Thread(
                target=self._run_emergency_mode,
                args=(lane_id,),
                daemon=True
            )
            emergency_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate emergency mode: {e}")
            return False

    def _run_emergency_mode(self, emergency_lane):
        """Run emergency mode for specified duration"""
        try:
            
            emergency_duration = 30
            
            for remaining in range(emergency_duration, 0, -1):
                if not self.emergency_active:
                    return
                time.sleep(1)
            
            
            self.deactivate_emergency_mode()
            
        except Exception as e:
            logger.error(f"Emergency mode execution error: {e}")
            self.emergency_active = False

    def deactivate_emergency_mode(self):
        """Deactivate emergency mode"""
        try:
            self.emergency_active = False
            self.timer_active = False
            self._set_all_lanes_red()
            self.current_green_lane = None
            
            
            for lane_id in self.lanes:
                self.lanes[lane_id]['priority'] = self._calculate_priority(self.lanes[lane_id]['vehicles'])
            
            self._log_event('EMERGENCY_DEACTIVATED', 'Emergency mode deactivated')
            
        
            socketio.emit('emergency_cleared', {
                'message': 'Emergency mode deactivated',
                'timestamp': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deactivate emergency mode: {e}")
            return False
    
    def update_lane_density(self, total_vehicles):
        """Distribute vehicles across 4 lanes"""
        try:
            if total_vehicles <= 0:
                for lane_id in self.lanes:
                    self.lanes[lane_id]['vehicles'] = 0
                    self.lanes[lane_id]['priority'] = 0
                return True
            
            
            patterns = [
                [0.35, 0.25, 0.25, 0.15],  
                [0.25, 0.30, 0.25, 0.20],  
                [0.25, 0.25, 0.35, 0.15],  
                [0.25, 0.25, 0.25, 0.25],  
            ]
            
            pattern = random.choice(patterns)
            random_factors = [random.uniform(0.8, 1.2) for _ in range(4)]
            total_factor = sum(pattern[i] * random_factors[i] for i in range(4))
            
            vehicles_assigned = 0
            for i in range(4):
                lane_id = i + 1
                if i < 3:
                    vehicle_ratio = (pattern[i] * random_factors[i]) / total_factor
                    lane_vehicles = max(0, int(total_vehicles * vehicle_ratio))
                else:
                    lane_vehicles = max(0, total_vehicles - vehicles_assigned)
                
                vehicles_assigned += lane_vehicles
                self.lanes[lane_id]['vehicles'] = lane_vehicles
                if not self.emergency_active:  # Don't override emergency priorities
                    self.lanes[lane_id]['priority'] = self._calculate_priority(lane_vehicles)
            
            self._log_event('DENSITY_UPDATE', f"Total: {total_vehicles}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update lane density: {e}")
            return False
    
    def _calculate_priority(self, vehicle_count):
        """Calculate lane priority"""
        if vehicle_count == 0:
            return 0
        elif vehicle_count <= 2:
            return 1
        elif vehicle_count <= 4:
            return 2
        elif vehicle_count <= 7:
            return 3
        else:
            return 4
    
    def find_highest_priority_lane(self):
        """Find lane with most vehicles"""
        if self.emergency_active:  
            return self.current_green_lane, self.lanes.get(self.current_green_lane)
        
        max_vehicles = max(lane['vehicles'] for lane in self.lanes.values())
        
        if max_vehicles == 0:
            return None, None
        
        for lane_id, lane_data in self.lanes.items():
            if lane_data['vehicles'] == max_vehicles:
                return lane_id, lane_data
        
        return None, None
    
    def calculate_green_time(self, vehicle_count):
        """Calculate green light duration"""
        if self.emergency_active:
            return 30  
        
        if vehicle_count == 0:
            return self.min_green_time
        
        calculated_time = self.min_green_time + (vehicle_count * self.base_time_per_vehicle)
        return max(self.min_green_time, min(calculated_time, self.max_green_time))
    
    def start_traffic_cycle(self):
        """Start traffic signal cycle"""
        if self.cycle_active or self.emergency_active:  # Don't start during emergency
            return False
        
        try:
            priority_lane_id, priority_lane = self.find_highest_priority_lane()
            
            if priority_lane_id is None:
                return False
            
            green_duration = self.calculate_green_time(priority_lane['vehicles'])
            
            self.cycle_active = True
            self.current_green_lane = priority_lane_id
            
            self._log_event('CYCLE_START', {
                'lane': priority_lane_id,
                'vehicles': priority_lane['vehicles'],
                'duration': green_duration
            })
            
           
            cycle_thread = threading.Thread(
                target=self._run_signal_cycle,
                args=(priority_lane_id, green_duration),
                daemon=True
            )
            cycle_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start cycle: {e}")
            self.cycle_active = False
            return False
    
    def _run_signal_cycle(self, green_lane_id, green_duration):
        """Execute complete signal cycle"""
        try:
            # Phase 1: All red clearance
            self._set_all_lanes_red()
            time.sleep(self.all_red_time)
            
            if not self.cycle_active or self.emergency_active:  # Check for emergency
                return
            
         
            self.lanes[green_lane_id]['status'] = 'green'
            self.timer_active = True
            
            for remaining in range(green_duration, 0, -1):
                if not self.cycle_active or self.emergency_active:  # Emergency override
                    return
                time.sleep(1)
            
           
            if not self.emergency_active:
                self.lanes[green_lane_id]['status'] = 'yellow'
                time.sleep(self.yellow_time)
            
            if not self.cycle_active or self.emergency_active:
                return
            
         
            self._set_all_lanes_red()
            time.sleep(self.all_red_time)
            
        
            self.timer_active = False
            self.cycle_active = False
            self.current_green_lane = None
            
           
            self.lanes[green_lane_id]['vehicles'] = max(0, self.lanes[green_lane_id]['vehicles'] - 2)
            
            self._log_event('CYCLE_COMPLETE', f"Lane {green_lane_id} cycle finished")
            
        except Exception as e:
            logger.error(f"Signal cycle error: {e}")
            self.cycle_active = False
            self.timer_active = False
    
    def _set_all_lanes_red(self):
        """Set all lanes to red"""
        for lane_id in self.lanes:
            self.lanes[lane_id]['status'] = 'red'
    
    def stop_cycle(self):
        """Stop current cycle"""
        if self.cycle_active:
            self.cycle_active = False
            self.timer_active = False
            self._set_all_lanes_red()
            if not self.emergency_active:
                self.current_green_lane = None
            self._log_event('CYCLE_STOPPED', 'Emergency stop')
            return True
        return False
    
    def get_current_status(self):
        """Get current system status"""
        return {
            'lanes': self.lanes,
            'current_green_lane': self.current_green_lane,
            'timer_active': self.timer_active,
            'cycle_active': self.cycle_active,
            'emergency_active': self.emergency_active,  # Added emergency status
            'timestamp': datetime.now().isoformat()
        }
    
    def get_lane_summary(self):
        """Get lane summary"""
        return {f"Lane_{lane_id}": {
            'name': lane_data['name'],
            'vehicles': lane_data['vehicles'],
            'status': lane_data['status'],
            'priority': lane_data['priority']
        } for lane_id, lane_data in self.lanes.items()}
    
    def get_signal_log(self, limit=10):
        """Get recent signal log"""
        return self.signal_log[-limit:] if self.signal_log else []
    
    def _log_event(self, action, details=""):
        """Log system events"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }
        self.signal_log.append(log_entry)
        logger.info(f"{action}: {details}")

traffic_manager = TrafficLaneManager()



@socketio.on('connect')
def handle_connect():
    print("🔗 Client connected to admin panel")
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print("🔌 Client disconnected from admin panel")

@socketio.on('user_sos_signal')
def handle_user_sos(data):
    """Handle SOS signal from user app"""
    print(f"🚨 EMERGENCY SOS RECEIVED: {data}")
    
    emergency_data = {
        'id': len(emergency_signals) + 1,
        'timestamp': datetime.now().isoformat(),
        'user': data.get('user', 'Unknown'),
        'location': data.get('location', 'Unknown'),
        'emergency_type': data.get('emergency_type', 'General Emergency'),
        'coordinates': data.get('coordinates', 'Unknown'),
        'phone': data.get('phone', 'Not provided'),
        'status': 'active'
    }
    
    emergency_signals.append(emergency_data)
    
    # Activate emergency mode (default to lane 1)
    success = traffic_manager.activate_emergency_mode(lane_id=1)
    
    if success:
        socketio.emit('emergency_alert', {
            'type': 'sos_received',
            'data': emergency_data,
            'message': f'🚨 EMERGENCY: {data.get("emergency_type")} at {data.get("location")}'
        })
        
        emit('sos_confirmation', {
            'status': 'success',
            'message': 'Emergency signal received and processed',
            'emergency_id': emergency_data['id']
        })
        
        print(f"✅ Emergency mode activated for SOS ID: {emergency_data['id']}")
    else:
        emit('sos_confirmation', {
            'status': 'error',
            'message': 'Failed to activate emergency mode'
        })
        print("❌ Failed to activate emergency mode")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def fallback_detection(filepath, filename):
    """Fallback detection when YOLO unavailable"""
    import shutil
    vehicle_count = random.randint(1, 8)
    
    try:
        output_filename = f"detected_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        shutil.copy2(filepath, output_path)
        return vehicle_count, output_filename
    except:
        return vehicle_count, filename

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main page"""
    try:
        return render_template('index.html')
    except:
        return jsonify({'status': 'error', 'message': 'Template not found'}), 500

@app.route('/health')
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'message': 'Traffic Management System running',
        'yolo_available': YOLO_AVAILABLE,
        'users': len(users_db),
        'emergency_signals': len(emergency_signals),  # Added emergency info
        'emergency_active': traffic_manager.emergency_active,
        'timestamp': datetime.now().isoformat()
    })

# Authentication Routes
@app.route('/register', methods=['POST'])
def register():
    """User registration"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'status': 'error', 'message': 'Username and password required'}), 400
        
        if len(username) < 3 or len(password) < 4:
            return jsonify({'status': 'error', 'message': 'Username min 3 chars, password min 4 chars'}), 400
        
        if username in users_db:
            return jsonify({'status': 'error', 'message': 'Username already exists'}), 400
        
        users_db[username] = password  # Simple storage
        return jsonify({'status': 'success', 'message': 'User registered!'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Registration failed'}), 500

@app.route('/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'status': 'error', 'message': 'Username and password required'}), 400
        
        if users_db.get(username) == password:
            return jsonify({'status': 'success', 'role': 'User'})
        
        return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Login failed'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': 'Invalid file'}), 400
        
        import uuid
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        if YOLO_AVAILABLE:
            try:
                total_vehicles, output_filename = detect_vehicles(filepath, save_output=True)
                method = "YOLO Detection"
            except:
                total_vehicles, output_filename = fallback_detection(filepath, unique_filename)
                method = "Fallback Detection"
        else:
            total_vehicles, output_filename = fallback_detection(filepath, unique_filename)
            method = "Fallback Detection"
        
        return jsonify({
            'status': 'success',
            'message': method,
            'total': total_vehicles,
            'image_url': f"/uploads/{output_filename or unique_filename}"
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Processing failed: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except:
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

# Lane Management Routes
@app.route('/api/lanes/update', methods=['POST'])
def update_lanes():
    """Update lane densities"""
    try:
        data = request.get_json()
        total_vehicles = data.get('total_vehicles', 0)
        
        if total_vehicles <= 0:
            return jsonify({'status': 'error', 'message': 'No vehicles to process'}), 400
        
        success = traffic_manager.update_lane_density(total_vehicles)
        if not success:
            return jsonify({'status': 'error', 'message': 'Failed to update lanes'}), 500
        
        cycle_started = False
        if not traffic_manager.emergency_active:  # Don't start during emergency
            cycle_started = traffic_manager.start_traffic_cycle()
        
        return jsonify({
            'status': 'success',
            'message': 'Lane analysis complete',
            'data': {
                'lanes': traffic_manager.get_lane_summary(),
                'cycle_started': cycle_started,
                'current_status': traffic_manager.get_current_status()
            }
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Update failed: {str(e)}'}), 500

@app.route('/api/lanes/status', methods=['GET'])
def get_lanes_status():
    """Get current lane status"""
    return jsonify({
        'status': 'success',
        'data': traffic_manager.get_current_status()
    })

@app.route('/api/lanes/stop', methods=['POST'])
def stop_traffic_cycle():
    """Stop traffic cycle"""
    try:
        stopped = traffic_manager.stop_cycle()
        return jsonify({
            'status': 'success',
            'message': 'Cycle stopped' if stopped else 'No active cycle',
            'stopped': stopped
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Stop failed'}), 500

@app.route('/api/lanes/log', methods=['GET'])
def get_signal_log():
    """Get signal log"""
    try:
        limit = request.args.get('limit', 20, type=int)
        return jsonify({
            'status': 'success',
            'data': {
                'log_entries': traffic_manager.get_signal_log(limit),
                'total_entries': len(traffic_manager.signal_log)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Log retrieval failed'}), 500

# ============================================================================
# SOS API ROUTES (Added)
# ============================================================================

@app.route('/api/emergency/sos', methods=['POST'])
def handle_emergency_sos():
    """Handle emergency SOS via HTTP API"""
    try:
        data = request.get_json()
        user_data = data.get('user_data', {})
        
        emergency_data = {
            'id': len(emergency_signals) + 1,
            'timestamp': datetime.now().isoformat(),
            'user': user_data.get('user', 'Unknown'),
            'location': user_data.get('location', 'Unknown'),
            'emergency_type': user_data.get('emergency_type', 'General Emergency'),
            'coordinates': user_data.get('coordinates', 'Unknown'),
            'phone': user_data.get('phone', 'Not provided'),
            'status': 'active',
            'method': 'HTTP API'
        }
        
        emergency_signals.append(emergency_data)
        
        success = traffic_manager.activate_emergency_mode(1)
        
        if success:
            socketio.emit('emergency_alert', {
                'type': 'sos_received',
                'data': emergency_data,
                'message': f'🚨 EMERGENCY via API: {user_data.get("emergency_type")} at {user_data.get("location")}'
            })
            
            return jsonify({
                'status': 'success',
                'message': 'Emergency SOS processed successfully',
                'emergency_id': emergency_data['id']
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to activate emergency mode'
            }), 500
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Emergency processing failed: {str(e)}'}), 500

@app.route('/api/emergency/simple', methods=['POST'])
def handle_simple_emergency():
    """Handle simple emergency POST"""
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        emergency_data = {
            'id': len(emergency_signals) + 1,
            'timestamp': datetime.now().isoformat(),
            'user': data.get('user', 'Unknown'),
            'location': data.get('location', 'Unknown'),
            'emergency_type': data.get('type', 'General Emergency'),
            'status': 'active',
            'method': 'Simple POST'
        }
        
        emergency_signals.append(emergency_data)
        
        success = traffic_manager.activate_emergency_mode(1)
        
        return jsonify({
            'status': 'success' if success else 'error',
            'message': 'Emergency received' if success else 'Failed to process',
            'emergency_id': emergency_data['id']
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/emergency/clear', methods=['POST'])
def clear_emergency():
    """Clear emergency mode"""
    try:
        success = traffic_manager.deactivate_emergency_mode()
        
        if success:
            for signal in emergency_signals:
                if signal['status'] == 'active':
                    signal['status'] = 'resolved'
                    signal['resolved_at'] = datetime.now().isoformat()
            
            return jsonify({
                'status': 'success',
                'message': 'Emergency mode cleared successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to clear emergency mode'
            }), 500
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Error Handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'status': 'error', 'message': 'File too large (max 200MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'status': 'error', 'message': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# ============================================================================
# MAIN APPLICATION RUNNER (Modified for SocketIO)
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Traffic Management System...")
    print("📁 Upload folder:", app.config['UPLOAD_FOLDER'])
    print("🌐 Server: http://127.0.0.1:5001")
    print("📊 Health: http://127.0.0.1:5001/health")
    print("🚦 YOLO:", "✅ Available" if YOLO_AVAILABLE else "❌ Fallback mode")
    print("🚨 SOS System: Ready")  # Added SOS indicator
    
    socketio.run(app, debug=True, host='127.0.0.1', port=5001)  # Changed to socketio.run
