import requests
import numpy as np
from typing import Optional
from langchain_core.tools import tool

@tool
def devices_check(url: Optional[str] = 'https://iomt.hoangphucthanh.vn:3030/iot') -> str:
    """
    Description: Checks the operational status of all IoT devices and reports any electrical abnormalities.
    Detects four types of electrical issues: over voltage, over current, over power, and under voltage.
    
    Args:
        url: (Optional) The API endpoint URL to fetch device status. Defaults to the production IOMT server.
              Should follow the format: 'https://[domain]:[port]/iot'
    
    Returns:
        str: A clear status report with one of these formats:
             - "All devices are working well. Don't worry" (if no issues)
             - A detailed list of devices with problems, specifying the exact electrical issue for each
    
    Example Response:
        "The problem(s) of system is (are):
            auo-display is having problem with over voltage
            led-nova is having problem with under voltage"
    """
    devices = ['auo-display', 'camera-control', 'electronic', 'led-nova']
    status = ['over voltage', 'over current', 'over power', 'under voltage']
    devices_status = np.empty((4,4), dtype = bool)

    for i in range(4):
        response = requests.get(f'{url}/{devices[i]}/latest')
        data = response.json()
        over_volt_status = data['data']['over_voltage_operating']
        over_cur_status = data['data']['over_current_operating']
        over_pow_status = data['data']['over_power_operating']
        under_volt_status = data['data']['under_voltage_operating']
        devices_status[i, :] = np.array([over_volt_status, over_cur_status, over_pow_status, under_volt_status])

    abnormal = np.argwhere(devices_status)
    if len(abnormal) == 0:
        return "All devices are work well. Don't worry"
    else:
        text = "The problem(s) of system is (are):\n"
        for i, j in abnormal:
            text = text + f"\t {devices[i]} is having problem with {status[j]} \n"
        return text

@tool    
def environment_check(url: Optional[str] = 'https://iomt.hoangphucthanh.vn:3030/iot/iot-env/latest') -> str:
    """
    Description: Monitors environmental conditions in the room, specifically temperature and humidity levels.
    Alerts when values exceed safe operating thresholds.
    
    Args:
        url: (Optional) The API endpoint URL for environmental data. Defaults to production IOMT server.
              Should follow the format: 'https://[domain]:[port]/iot/iot-env/latest'
    
    Returns:
        str: A status report with one of these formats:
             - "The environment is good. Don't worry" (if all conditions normal)
             - Clear warning about high temperature and/or humidity if detected
    
    Example Response:
        "The problem(s) with environment is (are):
            High temperature
            High humidity"
    """
    conditions = ['over_temperature', 'over_humidity']
    conditions_status = []

    response = requests.get(f"{url}")
    data = response.json()
    data = data['data']
    
    for cond in conditions:
        cond_status = data[cond]
        conditions_status.append(cond_status)
    conditions_status = np.array(conditions_status)

    abnormal = np.argwhere(conditions_status)
    if len(abnormal) == 0:
        return "The envrironment is good. Don't worry"
    else:
        text = "The problem(s) with environment is (are):"
        if conditions_status[0] == True:
            text = text + f"\t High temperature \n"
        if conditions_status[1] == True:
            text = text + f"\t High humidity \n"
        return text

@tool
def leakage_current_check(url: Optional[str] = 'https://iomt.hoangphucthanh.vn:3030/iot/iot-env/latest') -> str:
    """
    Description: Detects and reports electrical leakage current with three severity levels:
    1. Soft warning (minor leakage)
    2. Strong warning (significant leakage requiring immediate attention)
    3. Shutdown warning (dangerous leakage requiring emergency shutdown)
    
    Args:
        url: (Optional) The API endpoint URL for leakage data. Defaults to production IOMT server.
              Should follow the format: 'https://[domain]:[port]/iot/iot-env/latest'
    
    Returns:
        str: A status report with one of these formats:
             - "There is no leakage current. Don't worry" (if no leakage)
             - Clear warning with appropriate urgency level and recommended actions
    
    Example Responses:
        "The problem(s) with environment is (are):
            The current is a little bit leakage."
            
        "The problem(s) with environment is (are):
            The current is leakage terribly. Shut down the devices and call technician right now!"
    """
    leakies = ['soft_warning', 'strong_warning', 'shutdown_warning']
    leaky_status = []

    response = requests.get(f"{url}")
    data = response.json()
    data = data['data']
    
    for leak in leakies:
        leak_status = data[leak]
        leaky_status.append(leak_status)
    leaky_status = np.array(leaky_status)

    abnormal = np.argwhere(leaky_status)
    if len(abnormal) == 0:
        return "There is no leakage current. Don't worry"
    else:
        text = "The problem(s) with environment is (are):"
        if leaky_status[0] == True:
            text = text + f"\t The current is a litte bit leakage. \n"
        elif leaky_status[1] == True:
            text = text + f"\t The current is leakage. Check the socket and fix it imediately \n"
        else:
            text = text + f"\t The current is leakage terribly. Shut down the devices and call technician right now! \n"
        return text
