#!/usr/bin/env python3
import json

# Load existing ontology
d = json.load(open('/home/gio/dev/physis-pro/physis-core/config/engineering_ontology.json'))
domains = d.setdefault('domains', [])

# New engineering domains to add (physical/operational focus)
new_domains = [
    # Manufacturing & Fabrication
    {"name": "Pump Maintenance", "category": "Mechanical Engineering", "domain": "HEAL", "mode": "WORK", "axis_kind": "engineering", "axis_name": "repair", "unit": "scopes", "hints": ["impeller", "seal", "bearing", "alignment"]},
    {"name": "Valve Regulation", "category": "Fluid Engineering", "domain": "FABRICATE", "mode": "OPERATE", "axis_kind": "engineering", "axis_name": "flow_control", "unit": "calibrations", "hints": ["setpoint", "PV", "SV", "valve_position"]},
    {"name": "Conveyor Belt Tracking", "category": "Material Handling", "domain": "STUDY", "mode": "SENSE", "axis_kind": "engineering", "axis_name": "alignment", "unit": "sessions", "hints": ["tracking", "sensor", "feedback", "tension"]},
    {"name": "Extruder Temperature Tuning", "category": "polymer Engineering", "domain": "FABRICATE", "mode": "PLAN", "axis_kind": "engineering", "axis_name": "temperature_profile", "unit": "runs", "hints": ["melt_temp", "nozzle", "die swell", "cooling"]},
    {"name": "Pressurized Vessel Inspection", "category": "Safety Engineering", "domain": "HEAL", "mode": "WORK", "axis_kind": "engineering", "axis_name": "integrity_check", "unit": "inspections", "hints": ["PDO", "NDT", "pressure_test", "safety_valve"]},
    {"name": "Motor Alignment", "category": "Mechanical Engineering", "domain": "HEAL", "mode": "WORK", "axis_kind": "engineering", "axis_name": "mechanical_fit", "unit": "alignments", "hints": ["coupling", "shaft", "bearing", "laser"]},
    {"name": "PID Loop Tuning", "category": "Control Systems", "domain": "FABRICATE", "mode": "OPERATE", "axis_kind": "engineering", "axis_name": "gain_scheduling", "unit": "tuning_sessions", "hints": ["Kp", "Ki", "Kd", "setpoint", "overshoot"]},
    {"name": "Safety Relay Testing", "category": "Functional Safety", "domain": "HEAL", "mode": "WORK", "axis_kind": "engineering", "axis_name": "contact_check", "unit": "tests", "hints": ["norm_open", "norm_close", "test_pulse", "response_time"]},
    {"name": "Weld Quality Inspection", "category": "Manufacturing Engineering", "domain": "STUDY", "mode": "SENSE", "axis_kind": "engineering", "axis_name": "penetration", "unit": "samples", "hints": ["radiographic", "ultrasonic", "visual", "defect"]},
    {"name": "Pump Startup Sequence", "category": "Mechanical Engineering", "domain": "OPERATE", "mode": "LIFT", "axis_kind": "engineering", "axis_name": "procedure", "unit": "sequences", "hints": ["priming", "venting", "gradual_accel", "monitoring"]},

    # Chemical Engineering
    {"name": "Reactor Pressure Control", "category": "Chemical Engineering", "domain": "FABRICATE", "mode": "OPERATE", "axis_kind": "engineering", "axis_name": "pressure_regulation", "unit": "cycles", "hints": ["setpoint", "relief_valve", "PID", "monitoring"]},
    {"name": "Distillation Column Flooding", "category": "Chemical Engineering", "domain": "HEAL", "mode": "WORK", "axis_kind": "engineering", "axis_name": "flood_detection", "unit": "events", "hints": ["weeping", "entrainment", "pressure_spike", "level oscillation"]},

    # Civil/Infrastructure
    {"name": "Bridge Cable Inspection", "category": "Civil Engineering", "domain": "HEAL", "mode": "WALK", "axis_kind": "engineering", "axis_name": "cable_condition", "unit": "inspections", "hints": ["fatigue", "corrosion", "deflection", "measurement"]},
    {"name": "Road Surface Assessment", "category": "Civil Engineering", "domain": "STUDY", "mode": "SENSE", "axis_kind": "engineering", "axis_name": "rut_depth", "unit": "surveys", "hints": ["mapping", "depth", "texture", "grade"]},

    # Electrical Power
    {"name": "Transformer Oil Testing", "category": "Electrical Engineering", "domain": "HEAL", "mode": "WORK", "axis_kind": "engineering", "axis_name": "oil_quality", "unit": "tests", "hints": ["DGA", "moisture", "acid_number", "breakdown_voltage"]},
    {"name": "Cable Pull Tension", "category": "Electrical Engineering", "domain": "FABRICATE", "mode": "LIFT", "axis_kind": "engineering", "axis_name": "installation_tension", "unit": "pulls", "hints": ["pulling_length", "lubricant", "winch_setting", "clearance"]},

    # Industrial/Manufacturing
    {"name": "Robot Calibration", "category": "Manufacturing Engineering", "domain": "FABRICATE", "mode": "CREATE", "axis_kind": "engineering", "axis_name": "pose_accuracy", "unit": "calibrations", "hints": ["TCP", "joint_offset", "teach_pose", "repeatability"]},
    {"name": "Line Speed Optimization", "category": "Industrial Engineering", "domain": "OPERATE", "mode": "LIFT", "axis_kind": "engineering", "axis_name": "throughput", "unit": "increases", "hints": ["encoder", "gear_ratio", "motor_torque", "cycle_time"]},
]

# Extend domains list
current_count = len(domains)
for nd in new_domains:
    domains.append(nd)

print(f"Added {len(new_domains)} new domains. Total now: {current_count + len(new_domains)}")

# Save
d['domains'] = domains
with open('/home/gio/dev/physis-pro/physis-core/config/engineering_ontology.json', 'w') as f:
    json.dump(d, f, indent=2)
print("Saved successfully")