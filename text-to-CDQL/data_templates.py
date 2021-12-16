#Templates for data generation
templates = [
    {
        "text": "is the <A> in the <B> turning on?",
        "query": 'pull (device.hasState.isOn) define entity device is from smarthome where device.deviceName = "<A>" and  device.isIn.buildingenvironment = "<B>"'
    },

    {
        "text": "does the <A> in the <B> turn on?",
        "query": 'pull (device.hasState.isOn) define entity device is from smarthome where device.deviceName = "<A>" and  device.isIn.buildingenvironment = "<B>"'
    },
    {
        "text": "give me the state of the <A> in the <B>.",
        "query": 'pull (device.hasState.isOn) define entity device is from smarthome where device.deviceName = "<A>" and  device.isIn.buildingenvironment = "<B>"'
    },
    {
        "text": "in the <B>, give me the state of the <A>.",
        "query": 'pull (device.hasState.isOn) define entity device is from smarthome where device.deviceName = "<A>" and  device.isIn.buildingenvironment = "<B>"'
    },
    {
        "text": "in the <B>, is the <A> turning on.",
        "query": 'pull (device.hasState.isOn) define entity device is from smarthome where device.deviceName = "<A>" and  device.isIn.buildingenvironment = "<B>"'
    },
    {
        "text": "does the <A> in the <B> turn on.",
        "query": 'pull (device.hasState.isOn) define entity device is from smarthome where device.deviceName = "<A>" and  device.isIn.buildingenvironment = "<B>"'
    },
    {
        "text": "What is the current <A> level in the <B>.",
        "query": 'pull (device.stateValue) define entity device is from smarthome where device.hasState.isOn = true and device.deviceName = "<A>" and device.isIn.buildingenvironment = "<B>"'
    },
    {
        "text": "Give me the <A> level in the <B>.",
        "query": 'pull (device.stateValue) define entity device is from smarthome where device.hasState.isOn = true and device.deviceName = "<A>" and device.isIn.buildingenvironment = "<B>"'
    },
    {
        "text": "in the <B>, What is the current <A> level.",
        "query": 'pull (device.stateValue) define entity device is from smarthome where device.hasState.isOn = true and device.deviceName = "<A>" and device.isIn.buildingenvironment = "<B>"'        
    },
    {
        "text": "<A> level in the <B>.",
        "query": 'pull (device.stateValue) define entity device is from smarthome where device.hasState.isOn = true and device.deviceName = "<A>" and device.isIn.buildingenvironment = "<B>"'        
    },
    {
        "text": "in the <B>, <A> level",
        "query": 'pull (device.stateValue) define entity device is from smarthome where device.hasState.isOn = true and device.deviceName = "<A>" and device.isIn.buildingenvironment = "<B>"'        
    }
]