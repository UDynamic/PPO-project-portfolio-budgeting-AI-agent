from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored

def status(level="info", msg="", parameter=None ):
    
    colors = {"info": "cyan",
              "warn": "yellow",
              "error": "red",
              "success": "green"}
    
    print(colored(f"\n[{level.upper()}]: {msg}\n", colors.get(level, "white")), parameter)

# 🧠 Random number helper (seedless by default)
def random_uniform(low=0.0, high=1.0):
    _random_uniform = np.random.uniform(low, high)
    status("success", "generated random uniform", _random_uniform)
    return _random_uniform

# 🎯 Random integer helper (seedless by default)
def random_uniform_int(low, high):
    _random_int = np.random.randint(low, high)
    status("success", "generated random integer", _random_int)
    return _random_int

def random_discrete(_low, _high, _step):
  
  _random_discrete = np.random.choice(np.arange(_low, _high, _step))
  status("success", "generated random discrete", _random_discrete)
  return _random_discrete

np.set_printoptions(precision=3, suppress=True)

RISK_LEVELS = ["LOW", "BALANCED", "HIGH"]
INFLATION_RATES = [random_uniform(0.02, 0.04), random_uniform(0.04, 0.06), random_uniform(0.06, 0.08)]
ROI_RANGES=[(0.05, 0.10), (0.10, 0.20), (0.25, 0.50)]


def generate_performance_configs(project_type="Normal"):
    match project_type:
        case "Normal":
            return {
                "COST_PERFORMANCE": random_uniform(0.75, 1.0),
                "SCHEDULE_PERFORMANCE": random_uniform(0.75, 1.0)
            }
        case "Low":
            return {
                "COST_PERFORMANCE": random_uniform(0.5, 0.75),
                "SCHEDULE_PERFORMANCE": random_uniform(0.5, 0.75)
            }
        case "High":
            return {
                "COST_PERFORMANCE": random_uniform(1.0, 1.25),
                "SCHEDULE_PERFORMANCE": random_uniform(1.0, 1.25)
            }
        case _:
            raise ValueError(f"Unknown project type: {project_type}")


CONFIGURATIONS= {
    "TOTAL_BUDGET": 1,
    "TOTAL_TIMESTEPS": 12,
    
    # 15%–30%, 5% STEPS
    "ADVANCED_PAYMENT_RATIO": random_discrete(0.15, 0.31, 0.05),  
    
    # 10%–15%, 5% STEPS
    "FINAL_PAYMENT_RATIO": random_discrete(0.10, 0.16, 0.05),
    
    "n_projects": random_uniform_int(5, 10),
}

INFLOW_COMPOSITIONS = {
    "LOW_RISK": {
        "milestone_based": 0.60,
        "ev_based":         0.25,
        "lump_sum":         0.15
    },
    "BALANCED": {
        "milestone_based": 0.50,
        "ev_based":         0.30,
        "lump_sum":         0.20
    },
    "HIGH_RISK": {
        "milestone_based": 0.40,
        "ev_based":         0.40,
        "lump_sum":         0.20
    }
}

LOW_RISK_SCENARIO_CONFIGS ={
    "RISK_LEVEL": RISK_LEVELS[0],
    "INFLATION_RATE": INFLATION_RATES[0],
    "ROI_RANGE": ROI_RANGES[0],
    "INFLOW_COMPOSITION": INFLOW_COMPOSITIONS["LOW_RISK"]
}

BALANCED_RISK_SCENARIO_CONFIGS ={
    "RISK_LEVEL": RISK_LEVELS[1],
    "INFLATION_RATE": INFLATION_RATES[1],
    "ROI_RANGE": ROI_RANGES[1],
    "INFLOW_COMPOSITION": INFLOW_COMPOSITIONS["BALANCED"]
}

HIGH_RISK_SCENARIO_CONFIGS ={
    "RISK_LEVEL": RISK_LEVELS[2],
    "INFLATION_RATE": INFLATION_RATES[2],
    "ROI_RANGE": ROI_RANGES[2],
    "INFLOW_COMPOSITION": INFLOW_COMPOSITIONS["HIGH_RISK"]
}


ACTIVE_SCENARIO = LOW_RISK_SCENARIO_CONFIGS
status("warn", "ACTIVE_SCENARIO", ACTIVE_SCENARIO)

def set_roi_rate(self):
    status("info", "Initiating set_roi_rate", "")
    """
    Set the ROI (Return on Investment) for a project based on the active scenario.
    
    ROI is randomly selected as a discrete multiple of 5% within the configured range.
    Uses the globally defined ACTIVE_SCENARIO dictionary.
    """
    low, high = ACTIVE_SCENARIO["ROI_RANGE"]


    step = 0.05

    # Generate discrete ROI choices, rounded to 2 decimals to avoid float artifacts
    roi_rate_choices = np.round(np.arange(low, high + step/2, step), 2)

    # Randomly pick one
    roi_rate_selected = float(np.random.choice(roi_rate_choices))

    # Assign clean float to instance
    ROI_RATE = round(roi_rate_selected, 2)
    
    self.ROI_RATE = ROI_RATE
    
    status("success", "generated and stored ROI_RATE", ROI_RATE)
    
    return ROI_RATE

def set_duration(self):
    status("info", "Initiating set_duration", "")

    t = CONFIGURATIONS["TOTAL_TIMESTEPS"]
    
    DURATION = random_uniform_int(t - 2, t + 3)
    
    self.DURATION = DURATION
    
    status("success", "generated and stored DURATION", DURATION)
    return DURATION

# random around 1
def set_bac(self):
    status("info", "Initiating set_bac", "")

    BAC = random_uniform(0.8, 1.2)
    
    self.BAC = BAC
    
    status("success", "generated and stored BAC", BAC)
    return BAC

def set_s_curve(self):
    status("info", "Initiating set_s_curve", "")

    # --- Extract base parameters ---
    DURATION = getattr(self, "DURATION", None)
    if DURATION is None:
        status("error", "DURATION couldn't be acquired", )
        status("warn", "Attemting set_duration", )
        DURATION = set_duration(self)  # ensure one exists'
    else :
        status("success", "DURATION acquired successfully", DURATION)

    BAC = getattr(self, "BAC", None)
    if BAC is None:
        status("error", "BAC couldn't be acquired", )
        status("warn", "Attemting set_bac", )
        BAC = set_bac(self)  # ensure one exists
    else:
        status("success", "BAC acquired successfully", BAC)
    
    # normalized time axis: use duration+1 points so curves align with your schedule array
    # t as Timestep
    t = np.arange(0, DURATION + 1)
    tau = np.linspace(0.0, 1.0, DURATION + 1)

    def s_front_loaded(tau, exponent=0.6, smooth=0.12):
        status("info", "Initiating s_front_loaded", "")
        # base concave shape
        y = tau ** exponent
        # mild logistic smoothing to enforce S-shape (move slightly toward sigmoid)
        y = (1 / (1 + np.exp(-( (y - 0.5) / smooth )) ) - 1/(1+np.exp(0.5/smooth)))  # center and shift
        # normalize to 0..1
        y = (y - y.min()) / (y.max() - y.min())
        
        status("success", "generated s_front_loaded", y)
        return y

    def s_balanced(tau, k=8.0, center=0.5):
        status("info", "Initiating s_balanced", "")
        
        y = 1.0 / (1.0 + np.exp(-k * (tau - center)))
        y = (y - y.min()) / (y.max() - y.min())
        
        status("success", "generated s_balanced", y)
        return y

    def s_back_loaded(tau, exponent=2.5, smooth=0.12):
        status("info", "Initiating s_back_loaded", "")

        y = tau ** exponent
        # mild logistic smoothing for S character
        y = (1 / (1 + np.exp(-((y - 0.5) / smooth))) - 1/(1+np.exp(0.5/smooth)))
        y = (y - y.min()) / (y.max() - y.min())
        
        status("success", "generated s_back_loaded", y)
        return y

    # dictionary (use your earlier selection logic)
    models = {
        "front": lambda: s_front_loaded(tau),
        "balanced":  lambda: s_balanced(tau),
        "back": lambda: s_back_loaded(tau)
    }

    # === Randomly pick a model ===
    random_curve_type = np.random.choice(list(models.keys()))
    model_func = models[random_curve_type]

    # === Generate normalized curve ===
    y = model_func()
    y /= y[-1]  # normalize to end at 1 (safety)

    # === Compute baseline BCWS and inflation adjustment ===
    SCURVE_FULL = BAC * y 
    
    # === Package as list of tuples ===
    SCURVE_LIST = np.column_stack((t.astype(int), SCURVE_FULL.astype(float)))
    SCURVE_LIST_NORMALIZED = np.array(SCURVE_LIST, copy=True)
    SCURVE_LIST_NORMALIZED[:, 1] /= float(BAC)

    # === Store attributes ===
    self.SCURVE_TYPE = random_curve_type
    self.SCURVE_FULL = SCURVE_FULL
    self.SCURVE_LIST = SCURVE_LIST
    self.SCURVE_LIST_NORMALIZED = SCURVE_LIST_NORMALIZED
    self.TIMESTEPS = t

    status("success", "generated and stored SCURVE_LIST", SCURVE_LIST)
    return SCURVE_LIST

def set_s_curve_periodic(self):
    status("info", "Initiating set_s_curve_periodic", "")

    SCURVE_LIST = getattr(self, "SCURVE_LIST", None)
    if SCURVE_LIST is None:
        status("error", "SCURVE_LIST couldn't be acquired", )
        status("warn", "Attemting set_s_curve", )
        SCURVE_LIST = set_s_curve(self)  # ensure one exists
    else :
        status("success", "SCURVE_LIST acquired successfully", SCURVE_LIST)
    
    SCURVE_LIST_PERIODIC = np.column_stack([SCURVE_LIST[0:, 0], np.insert(np.diff(SCURVE_LIST[:, 1]), 0, 0.0)])
    self.SCURVE_LIST_PERIODIC = SCURVE_LIST_PERIODIC
    status("info", "generated and stored SCURVE_LIST_PERIODIC", SCURVE_LIST_PERIODIC)
    
    return SCURVE_LIST_PERIODIC


def set_inflated_s_curve(self):
    status("info", "Initiating set_inflated_s_curve", "")

    # === (safe) importing attributes ===
    INFLATION_RATE = ACTIVE_SCENARIO["INFLATION_RATE"]
    if INFLATION_RATE is None:
        status("error", "INFLATION_RATE couldn't be acquired", )
    else:
        status("success", "INFLATION_RATE acquired successfully", INFLATION_RATE)

    DURATION = getattr(self, "DURATION", None)
    if DURATION is None:
        status("error", "DURATION couldn't be acquired", )
        status("warn", "Attemting set_duration", )
        DURATION = set_duration(self)  # ensure one exists'
    else :
        status("success", "DURATION acquired successfully", DURATION)

    SCURVE_LIST = getattr(self, "SCURVE_LIST", None)
    if SCURVE_LIST is None:
        status("error", "SCURVE_LIST couldn't be acquired", )
        status("warn", "Attemting set_s_curve", )
        SCURVE_LIST = set_s_curve(self)  # ensure one exists
    else :
        status("success", "SCURVE_LIST acquired successfully", SCURVE_LIST)

    # t as in Timesteps
    t = np.arange(0, DURATION + 1)

    # Generating the inflated s-curve
    LINEAR_INFLATION_LIST = 1 + INFLATION_RATE * (t / DURATION)
    self.LINEAR_INFLATION_LIST = LINEAR_INFLATION_LIST
    status("info", "generated and stored LINEAR_INFLATION_LIST", LINEAR_INFLATION_LIST)
    
    SCURVE_LIST_INFLATION_ADJUSTED = np.array(SCURVE_LIST, copy=True)
    SCURVE_LIST_INFLATION_ADJUSTED[:, 1] *= LINEAR_INFLATION_LIST
    self.SCURVE_LIST_INFLATION_ADJUSTED = SCURVE_LIST_INFLATION_ADJUSTED
    status("success", "generated and stored SCURVE_LIST_INFLATION_ADJUSTED", SCURVE_LIST_INFLATION_ADJUSTED)
    
    return SCURVE_LIST_INFLATION_ADJUSTED
  
def set_expected_payments_list(self):
    status("info", "Initiating set_expected_payments_list", "")
 
    # --- Extract base parameters ---
    ADVANCED_PAYMENT_RATIO = CONFIGURATIONS["ADVANCED_PAYMENT_RATIO"]
    if ADVANCED_PAYMENT_RATIO is None:
        status("error", "ADVANCED_PAYMENT_RATIO couldn't be acquired", )
    else:
        status("success", "ADVANCED_PAYMENT_RATIO acquired successfully", ADVANCED_PAYMENT_RATIO)
        
    FINAL_PAYMENT_RATIO   = CONFIGURATIONS["FINAL_PAYMENT_RATIO"]
    if FINAL_PAYMENT_RATIO is None:
        status("error", "FINAL_PAYMENT_RATIO couldn't be acquired", )
    else:
        status("success", "FINAL_PAYMENT_RATIO acquired successfully", FINAL_PAYMENT_RATIO)

    DURATION = getattr(self, "DURATION", None)
    if DURATION is None:
        status("error", "DURATION couldn't be acquired", )
        status("warn", "Attemting set_duration", )
        DURATION = set_duration(self)  # ensure one exists'
    else :
        status("success", "DURATION acquired successfully", DURATION)

    BAC = getattr(self, "BAC", None)
    if BAC is None:
        status("error", "BAC couldn't be acquired", )
        status("warn", "Attemting set_bac", )
        BAC = set_bac(self)  # ensure one exists
    else:
        status("success", "BAC acquired successfully", BAC)

    SCURVE_FULL = getattr(self, "SCURVE_FULL", None)
    if SCURVE_FULL is None:
        status("error", "SCURVE_FULL couldn't be acquired", )
        status("warn", "Attemting ****", )
        # SCURVE_FULL = (self)  # ensure one exists
    else: status("success", "SCURVE_FULL acquired successfully", SCURVE_FULL)
    
    SCURVE_LIST = getattr(self, "SCURVE_LIST", None)
    if SCURVE_LIST is None:
        status("error", "SCURVE_LIST couldn't be acquired", )
        status("warn", "Attemting set_s_curve", )
        SCURVE_LIST = set_s_curve(self)  # ensure one exists
    else: status("success", "SCURVE_LIST acquired successfully", SCURVE_LIST)
    
    SCURVE_LIST_NORMALIZED = getattr(self, "SCURVE_LIST_NORMALIZED", None)
    if SCURVE_LIST_NORMALIZED is None:
        status("error", "SCURVE_LIST_NORMALIZED couldn't be acquired", )
        status("warn", "Attemting ****", )
        # SCURVE_LIST_NORMALIZED = (self)  # ensure one exists
    else: status("success", "SCURVE_LIST_NORMALIZED acquired successfully", SCURVE_LIST_NORMALIZED)
    
    INFLATION_RATE = ACTIVE_SCENARIO["INFLATION_RATE"]
    if INFLATION_RATE is None:
        status("error", "INFLATION_RATE couldn't be acquired", )
        status("warn", "Attemting ****", )
        # INFLATION_RATE = (self)  # ensure one exists
    else: status("success", "INFLATION_RATE acquired successfully", INFLATION_RATE)
    
    LINEAR_INFLATION_LIST = getattr(self, "LINEAR_INFLATION_LIST", None)
    if LINEAR_INFLATION_LIST is None:
        status("error", "LINEAR_INFLATION_LIST couldn't be acquired", )
        status("warn", "Attemting ****", )
        # LINEAR_INFLATION_LIST = (self)  # ensure one exists
    else: status("success", "LINEAR_INFLATION_LIST acquired successfully", LINEAR_INFLATION_LIST)
    
    SCURVE_LIST_INFLATION_ADJUSTED = getattr(self, "SCURVE_LIST_INFLATION_ADJUSTED", None)
    if SCURVE_LIST_INFLATION_ADJUSTED is None:
        status("error", "SCURVE_LIST_INFLATION_ADJUSTED couldn't be acquired", )
        status("warn", "Attemting ****", )
        # SCURVE_LIST_INFLATION_ADJUSTED = (self)  # ensure one exists
    else: status("success", "SCURVE_LIST_INFLATION_ADJUSTED acquired successfully", SCURVE_LIST_INFLATION_ADJUSTED)

    ROI_RATE = getattr(self, "ROI_RATE", None)
    if ROI_RATE is None:
        status("error", "ROI_RATE couldn't be acquired", )
        status("warn", "Attemting set_roi_rate", )
        ROI_RATE = set_roi_rate(self)  # ensure one exists
    else: status("success", "ROI_RATE acquired successfully", ROI_RATE)
    
    SCURVE_LIST_PERIODIC = getattr(self, "SCURVE_LIST_PERIODIC", None)
    if SCURVE_LIST_PERIODIC is None:
        status("error", "SCURVE_LIST_PERIODIC couldn't be acquired", )
        status("warn", "Attemting set_s_curve_periodic", )
        SCURVE_LIST_PERIODIC = set_s_curve_periodic(self)  # ensure one exists
    else: status("success", "SCURVE_LIST_PERIODIC acquired successfully", SCURVE_LIST_PERIODIC)



    t = np.arange(0, DURATION + 1)
    
    # adjusting s-curve to the ROI rate
    _scale_factor = (1.0 + ROI_RATE)
    _total_expected_payment = BAC * _scale_factor
    self.total_expected_payment = _total_expected_payment
    status("success", "generated and stored total_expected_payment", _total_expected_payment)
    
    SCURVE_LIST_ROI_ADJUSTED_PERIODIC = np.array(SCURVE_LIST_PERIODIC, copy=True)
    SCURVE_LIST_ROI_ADJUSTED_PERIODIC[:, 1] *= _scale_factor
    self.SCURVE_LIST_ROI_ADJUSTED_PERIODIC = SCURVE_LIST_ROI_ADJUSTED_PERIODIC
    status("success", "generated and stored SCURVE_LIST_ROI_ADJUSTED_PERIODIC", SCURVE_LIST_ROI_ADJUSTED_PERIODIC)
    
    SCURVE_LIST_ROI_ADJUSTED = np.column_stack([t.astype(int), np.cumsum(SCURVE_LIST_ROI_ADJUSTED_PERIODIC[:, 1]).astype(float)])
    self.SCURVE_LIST_ROI_ADJUSTED = SCURVE_LIST_ROI_ADJUSTED
    status("success", "generated and stored SCURVE_LIST_ROI_ADJUSTED", SCURVE_LIST_ROI_ADJUSTED)


    _advanced_payment = float(ADVANCED_PAYMENT_RATIO) * _total_expected_payment
    _final_payment = float(FINAL_PAYMENT_RATIO) * _total_expected_payment
    _before_final_payment = _total_expected_payment - _final_payment
    
    # === 1️⃣ Milestone-Based Model ===
    def milestone_based():
        status("info", "Initiating milestone_based", "")

        # --- Intermediate milestone positions ---
        # the first timestep from the end in which the s-curve value is lass than or equal to  _before_final_payment
        for i in reversed(t):
            if SCURVE_LIST_ROI_ADJUSTED[i, 1] < _before_final_payment:
                _last_intermediate_possible_milestone_boundry = i + 1
                break
        
        _possible_positions = np.arange(1, _last_intermediate_possible_milestone_boundry)
        _n_middle = random_uniform_int(1, min(4, len(_possible_positions)))
        _middle_positions = sorted(np.random.choice(_possible_positions, _n_middle, replace=False))
        _milestone_positions = [0] + _middle_positions + [DURATION]
        self.milestone_positions = _milestone_positions
        status("success", "generated and stored milestone_positions", _milestone_positions)
        
        _milestone_based_payment_schedule = np.column_stack([t.astype(int), np.zeros_like(t).astype(float)])
        _milestone_based_payment_schedule[-1, 1] = _total_expected_payment  # final payment known

        # --- Work backwards from final payment ---
        __cumulative_ramining = _before_final_payment
        j = DURATION
        for _pos in reversed(_middle_positions):
            # Compute how much should have been paid up to this milestone
            # based on BCWS share relative to total cost curve
            for i in range(_pos, j):  
                _milestone_based_payment_schedule[i, 1] = __cumulative_ramining
                j = _pos
            
            if (SCURVE_LIST_ROI_ADJUSTED[_pos, 1] > _advanced_payment):
              __cumulative_ramining = SCURVE_LIST_ROI_ADJUSTED[_pos, 1]
            else: 
                __cumulative_ramining = _advanced_payment
                
        # --- Advance payment (first payment at t=0) ---
        for i in range(0, _middle_positions[0]):
            _milestone_based_payment_schedule[i, 1] = _advanced_payment
        
           
        self.milestone_based_payment_schedule = _milestone_based_payment_schedule
        status("success", "generated and stored milestone_based_payment_schedule", _milestone_based_payment_schedule)
        return _milestone_based_payment_schedule


    # === 2️⃣ EV-Based Model ===
    def ev_based():
        status("info", "Initiating ev_based", "")
        
        _cum = _advanced_payment
        
        _ev_based_payment_schedule = np.column_stack([t.astype(int), np.zeros_like(t).astype(float)])
        _ev_based_payment_schedule[0, 1] = _cum

        for i in range(1, DURATION):
            _ev_based_payment_schedule[i, 1] = _cum
            if SCURVE_LIST_ROI_ADJUSTED[i, 1] > _cum:
                if (_before_final_payment) >= SCURVE_LIST_ROI_ADJUSTED[i, 1]: 
                    _cum = SCURVE_LIST_ROI_ADJUSTED[i, 1]
                _ev_based_payment_schedule[i, 1] = _cum

        # final payment adjustment
        _ev_based_payment_schedule[-1,-1] = _total_expected_payment

        self.ev_based_payment_schedule = _ev_based_payment_schedule
        status("success", "generated and stored ev_based_payment_schedule", _ev_based_payment_schedule)
        return _ev_based_payment_schedule

    # === 3️⃣ Lump-Sum Model ===
    def lump_sum():
        status("info", "Initiating lump_sum", "")
        """
        Single payment — either advance (at t=0) or final (at t=end),
        proportional to BAC and BCWS shape.
        """
        _lump_sum_payment_schedule = np.column_stack([t.astype(int), np.zeros_like(t).astype(float)])
        if random_uniform(0, 1) < 0.5:
            status("success", "generated random variable, Advanced payment as Lump sum payment", "")
            # Advance lump sum (based on BCWS fraction early)
            _lump_sum_payment_schedule[:, 1] = _total_expected_payment - _final_payment
            _lump_sum_payment_schedule[-1, 1] = _total_expected_payment
        else:
            status("success", "generated random variable, Final payment as Lump sum payment", "")
            # Final lump sum (based on BCWS end)
            _lump_sum_payment_schedule[-1, -1] = _total_expected_payment
        
        
        self.lump_sum_payment_schedule = _lump_sum_payment_schedule
        status("success", "generated and stored lump_sum_payment_schedule", _lump_sum_payment_schedule)
        return _lump_sum_payment_schedule


    # === Model selector ===
    models = {
        "milestone_based": milestone_based,
        "ev_based": ev_based,
        "lump_sum": lump_sum
    }

    def set_inflow_model(models, composition):
        status("info", "Initiating set_inflow_model", "")
        """
        models: dict -> {name: function}
        composition: dict -> {name: probability}
        """
        
        names = list(composition.keys())
        probs = list(composition.values())

        # numpy makes this simple
        chosen_name = np.random.choice(names, p=probs)

        status("info", "selected chosen_name", chosen_name)
        return chosen_name, models[chosen_name]
    
    _inflow_composition = ACTIVE_SCENARIO["INFLOW_COMPOSITION"]
    
    _inflow_type, chosen_func = set_inflow_model(models, _inflow_composition)
    
    EXPECTED_PAYMENTS_LIST = chosen_func()

    self.inflow_type = _inflow_type
    status("success", "generated and stored inflow_type", _inflow_type)
    
    self.EXPECTED_PAYMENTS_LIST = EXPECTED_PAYMENTS_LIST
    status("success", "generated and stored EXPECTED_PAYMENTS_LIST", EXPECTED_PAYMENTS_LIST)
    return EXPECTED_PAYMENTS_LIST


def set_project_datadate(self):
    status("info", "Initiating set_datadate", "")
    """
    Randomly assign a reporting 'data date' (timestep) within the project duration.
    """
    DURATION = getattr(self, "DURATION", None)
    if DURATION is None:
        status("error", "DURATION couldn't be acquired", )
        status("warn", "Attemting set_duration", )
        DURATION = set_duration(self)  # ensure one exists'
    else :
        status("success", "DURATION acquired successfully", DURATION)
        
    _project_datadate = random_uniform_int(0, self.DURATION + 1)
    self.project_datadate = _project_datadate
    status("success", "generated and stored datadate", _project_datadate)
    return _project_datadate

def set_performance(self):
    status("info", "Initiating set_performance", "")
    
    DURATION = getattr(self, "DURATION", None)
    if DURATION is None:
        status("error", "DURATION couldn't be acquired", )
        status("warn", "Attemting set_duration", )
        DURATION = set_duration(self)  # ensure one exists'
    else :
        status("success", "DURATION acquired successfully", DURATION)
        
    project_datadate = getattr(self, "project_datadate", None)
    if project_datadate is None:
        status("error", "project_datadate couldn't be acquired", )
        status("warn", "Attemting set_project_datadate", )
        project_datadate = set_project_datadate(self)  # ensure one exists'
    else :
        status("success", "project_datadate acquired successfully", project_datadate)


    SCURVE_LIST_ROI_ADJUSTED = getattr(self, "SCURVE_LIST_ROI_ADJUSTED", None)
    if SCURVE_LIST_ROI_ADJUSTED is None:
        status("error", "SCURVE_LIST_ROI_ADJUSTED couldn't be acquired", )
        status("warn", "Attemting ***", )
        # SCURVE_LIST_ROI_ADJUSTED = ***  # ensure one exists
    else: status("success", "SCURVE_LIST_ROI_ADJUSTED acquired successfully", SCURVE_LIST_ROI_ADJUSTED)
    
    SCURVE_LIST_ROI_ADJUSTED_PERIODIC = getattr(self, "SCURVE_LIST_ROI_ADJUSTED_PERIODIC", None)
    if SCURVE_LIST_ROI_ADJUSTED_PERIODIC is None:
        status("error", "SCURVE_LIST_ROI_ADJUSTED_PERIODIC couldn't be acquired", )
        status("warn", "Attemting ***", )
        # SCURVE_LIST_ROI_ADJUSTED_PERIODIC = ***  # ensure one exists
    else: status("success", "SCURVE_LIST_ROI_ADJUSTED_PERIODIC acquired successfully", SCURVE_LIST_ROI_ADJUSTED_PERIODIC)


    t = np.arange(0, DURATION + 1)

    # === BCWS ===
    _bcws = float(SCURVE_LIST_ROI_ADJUSTED[project_datadate, 1])
    status("success", "generated bcws", _bcws)


    # === BCWP ===
    _bcwp_list_periodic = np.column_stack([t.astype(int), np.zeros_like(t).astype(float)])
    for i in t:
        if i <= project_datadate:
            _bcwp_list_periodic[i, 1] = SCURVE_LIST_ROI_ADJUSTED_PERIODIC[i, 1] * generate_performance_configs()["SCHEDULE_PERFORMANCE"]
    status("success", "generated _bcwp_list_periodic", _bcwp_list_periodic)
    
    _bcwp_list = np.column_stack([t.astype(int), np.cumsum(_bcwp_list_periodic[:, 1]).astype(float)])
    status("success", "generated _bcwp_list", _bcwp_list)
    
    _bcwp = _bcwp_list[project_datadate, 1]
    status("success", "generated _bcwp", _bcwp)
    

    # === ACWP ===
    _acwp_list_periodic = np.column_stack([t.astype(int), np.zeros_like(t).astype(float)])
    for i in t:
        if i <= project_datadate:
            _acwp_list_periodic[i, 1] = SCURVE_LIST_ROI_ADJUSTED_PERIODIC[i, 1] * generate_performance_configs()["COST_PERFORMANCE"]
    status("success", "generated _acwp_list_periodic", _acwp_list_periodic)
    
    _acwp_list = np.column_stack([t.astype(int), np.cumsum(_acwp_list_periodic[:, 1]).astype(float)])
    status("success", "generated _acwp_list", _acwp_list)
    
    _acwp = _acwp_list[project_datadate, 1]
    status("success", "generated _acwp", _acwp)


    # === completion flag ===
    _project_progress = _bcwp / SCURVE_LIST_ROI_ADJUSTED[-1, 1]
    _project_plan = _bcws / SCURVE_LIST_ROI_ADJUSTED[-1, 1]
    
    _project_is_complete = _project_progress == 1.0

    # === performance indices ===
    _cpi = _bcwp / _acwp if _acwp != 0 else 0
    _spi = _bcwp / _bcws if _bcws != 0 else 0

    _project_performance = {
        "project_datadate": project_datadate,
        "project_progress": _project_progress,
        "project_plan": _project_plan,
        "bcws": _bcws,
        "bcwp_list_periodic": _bcwp_list_periodic,
        "bcwp_list": _bcwp_list,
        "bcwp": _bcwp,
        "acwp_list_periodic": _acwp_list_periodic,
        "acwp_list": _acwp_list,
        "acwp": _acwp,
        "cpi": _cpi,
        "spi": _spi,
        "project_is_complete": _project_is_complete
    }
    
    # === store and return all ===
    self.project_performance = _project_performance
    
    status("success", "generated and stored project_performance", _project_performance)
    return _project_performance

class ProjectClass:
    def __init__(self):
        self.ROI_RATE = set_roi_rate(self)
        self.DURATION = set_duration(self)
        self.BAC = set_bac(self)
        self.SCURVE_LIST  = set_s_curve(self)
        self.SCURVE_LIST_PERIODIC = set_s_curve_periodic(self)
        self.SCURVE_LIST_INFLATION_ADJUSTED = set_inflated_s_curve(self)
        self.EXPECTED_PAYMENTS_LIST = set_expected_payments_list(self)
        self.project_datadate = set_project_datadate(self)
        self.performance = set_performance(self)


def plot_project(ax, p):
    progress = p.performance

    t = np.arange(0, p.DURATION + 1)

    # --- ensure arrays for safe column access ---
    scurve = np.array(p.SCURVE_LIST_ROI_ADJUSTED)
    expected = np.array(p.EXPECTED_PAYMENTS_LIST)
    bcwp = np.array(progress["bcwp_list"])
    acwp = np.array(progress["acwp_list"])

    # === BCWS baseline ===
    ax.bar(t, scurve[:, 1], color="gold", alpha=0.8, label="BCWS (Baseline)")

    # === Expected Payments ===
    ax.step(t, expected[:, 1], where="post", linewidth=2, linestyle="dotted",
            marker="s", color="lime", label="Expected Cumulative Payments")

    # === BCWP ===
    ax.step(t, bcwp[:, 1], where="post", linewidth=2, linestyle="-",
            marker="o", color="aqua", label="BCWP")

    # === ACWP ===
    ax.step(t, acwp[:, 1], where="post", linewidth=2, linestyle="-",
            marker="^", color="teal", label="ACWP")

    # === Data Date ===
    ax.axvline(x=progress["project_datadate"], color="red", linestyle="--",
               linewidth=2, label=f"Data Date = {progress['project_datadate']}")

    # === Title, labels, and grid ===
    ax.set_title("Project Performance Snapshot", fontsize=12, weight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative Value (normalized to BAC)")
    ax.grid(True, linestyle=":", alpha=0.7)

    # === CPI and SPI text box ===
    metrics_text = f"CPI = {progress['cpi']:.2f}\nSPI = {progress['spi']:.2f}"
    ax.text(0.02, 0.85, metrics_text, transform=ax.transAxes,
            fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    ax.legend(loc="best", fontsize=7)


p = ProjectClass()

fig, ax = plt.subplots(figsize=(12, 6))
plot_project(ax, p)
plt.tight_layout()
plt.show()

class PortfolioClass:
    def __init__(self, n_projects):
        self.n_projects = n_projects
        self.projects = []
        self.generate_portfolio()

    def generate_portfolio(self):

        for _ in range(self.n_projects):
            p = ProjectClass()
            self.projects.append(p)
    
def plot_portfolio(portfolio):
    n = portfolio.n_projects
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten()  # flatten to a single list of axes

    for idx, project in enumerate(portfolio.projects):
        ax = axes[idx]
        plot_project(ax, project)
        ax.set_title(f"Project {idx+1}")

    # Hide unused axes (if any)
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


n_projects = CONFIGURATIONS["n_projects"]

portfolio = PortfolioClass(n_projects)
plot_portfolio(portfolio)
