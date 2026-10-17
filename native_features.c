/* Optional CPython helpers for the existing observation schema. No engine logic lives here. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

enum { BUFFS, MOVE, ATTACK_STOP, LEFT, SPEED, HIT, ATTACK, ABILITY, RETURN_SHOT, HIT_SPEED,
       KIND, REMAINING, FUSE, SPAWN, HIDDEN, IMMUNE, INVISIBLE, AIR, PUSHED, PUSH_LEFT, CYCLE,
       LOAD, CHARGE, DASH, PHASE, USED, SHIELD, SHIELD_MAX, STAGE, STATE, SINCE, USES, MAX_CHARGES,
       Z, JUMP, ELAPSED, TOTAL, SPECIAL, AGAIN, HOP, KEY_COUNT };
static const char *names[] = {
    "buffs", "stopsMovement", "stopsAttacks", "remainingMs", "speedPercent", "hitSpeedPercent",
    "attack", "ability", "returnShot", "hitSpeedMs", "kind", "remainingMs", "fuseMs", "spawnTimerMs",
    "hidden", "immune", "invisible", "airborne", "pushed", "pushRemainingMs", "cycleMs", "loadMs",
    "charge", "dashCooldownMs", "phase", "used", "shield", "shieldMax", "stage", "state", "sinceUseMs",
    "uses", "maxCharges", "z", "jumpRemainingMs", "elapsedMs", "totalMs", "specialCooldownMs",
    "invisibleAgainMs", "dashHop"
};
static PyObject *keys[KEY_COUNT];
static PyObject *item(PyObject *d, int k) {
    return d && PyDict_Check(d) ? PyDict_GetItemWithError(d, keys[k]) : NULL;
}
static double number(PyObject *d, int k) {
    PyObject *v = item(d, k);
    return !v || v == Py_None ? 0.0 : PyFloat_AsDouble(v);
}
static int truth(PyObject *d, int k) {
    PyObject *v = item(d, k);
    return v ? PyObject_IsTrue(v) : 0;
}
static double clip(double x, double lo, double hi) { return x < lo ? lo : x > hi ? hi : x; }
static double seconds(PyObject *d, int k, double cap) { return clip(number(d, k), 0, cap * 1000) / 1000; }

static PyObject *unit_values(PyObject *self, PyObject *u) {
    if (!PyDict_Check(u)) { PyErr_SetString(PyExc_TypeError, "unit must be a dict"); return NULL; }
    PyObject *attack = item(u, ATTACK), *ability = item(u, ABILITY), *buffs = item(u, BUFFS);
    int move = 0, stop = 0;
    double stopped = 0, speed = 0, hit = 0;
    if (buffs && buffs != Py_None) {
        PyObject *seq = PySequence_Fast(buffs, "buffs must be a sequence");
        if (!seq) return NULL;
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); ++i) {
            PyObject *b = PySequence_Fast_GET_ITEM(seq, i);
            int m = truth(b, MOVE), a = truth(b, ATTACK_STOP);
            move |= m; stop |= a;
            if (m || a) { double left = number(b, LEFT); if (left > stopped) stopped = left; }
            speed += number(b, SPEED); hit += number(b, HIT);
        }
        Py_DECREF(seq);
    }
    double hs = number(attack, HIT_SPEED), shield_max = number(u, SHIELD_MAX);
    double max_charges = number(ability, MAX_CHARGES), state = number(u, STATE);
    PyObject *kind = item(u, KIND), *phase = item(ability, PHASE), *shot = item(attack, RETURN_SHOT);
    int timer = SPAWN;
    if (kind && PyUnicode_Check(kind)) {
        if (PyUnicode_CompareWithASCIIString(kind, "area") == 0) timer = REMAINING;
        else if (PyUnicode_CompareWithASCIIString(kind, "timed") == 0) timer = FUSE;
    }
    double returning = 0;
    if (shot && PyObject_IsTrue(shot)) {
        double total = number(shot, TOTAL);
        if (total == 0) { PyErr_SetString(PyExc_ZeroDivisionError, "zero return-shot duration"); return NULL; }
        returning = clip(number(shot, ELAPSED) / total, 0, 1);
    }
    double out[31] = {
        truth(u,HIDDEN), truth(u,IMMUNE), truth(u,INVISIBLE), truth(u,AIR), move, stop,
        clip(stopped,0,5000)/1000, clip(speed/100,-1,2), clip(hit/100,-1,2),
        truth(u,PUSHED) ? seconds(u,PUSH_LEFT,5) : 0,
        hs > 0 ? clip(number(attack,CYCLE)/hs,0,1) : 0,
        seconds(attack,LOAD,5), clip(number(u,CHARGE)/10000,0,1), seconds(attack,DASH,5),
        phase && phase != Py_None, seconds(u,timer,10), 0, 0, truth(ability,USED),
        shield_max ? clip(number(u,SHIELD)/shield_max,0,1) : 0,
        clip(number(attack,STAGE),0,2)/2, state == 3, state == 5, seconds(ability,SINCE,10),
        max_charges ? clip(number(ability,USES)/max_charges,0,1) : 0,
        clip(number(u,Z)/3000,0,2), seconds(u,JUMP,2), returning, seconds(attack,SPECIAL,5),
        seconds(u,AGAIN,5), number(attack,HOP)/3 > 1 ? 1 : number(attack,HOP)/3
    };
    if (PyErr_Occurred()) return NULL;
    PyObject *result = PyTuple_New(31);
    if (!result) return NULL;
    for (int i = 0; i < 31; ++i) {
        PyObject *v = PyFloat_FromDouble(out[i]);
        if (!v) { Py_DECREF(result); return NULL; }
        PyTuple_SET_ITEM(result,i,v);
    }
    return result;
}

static PyObject *read_i32(PyObject *self, PyObject *arg) {
    void *p = PyLong_AsVoidPtr(arg); int32_t value;
    if (PyErr_Occurred()) return NULL;
    memcpy(&value,p,sizeof(value)); return PyLong_FromLong(value);
}
static PyObject *read_u64(PyObject *self, PyObject *arg) {
    void *p = PyLong_AsVoidPtr(arg); uint64_t value;
    if (PyErr_Occurred()) return NULL;
    memcpy(&value,p,sizeof(value)); return PyLong_FromUnsignedLongLong(value);
}
static PyMethodDef methods[] = {
    {"unit_state_values",unit_values,METH_O,"The existing 31 unit-state features."},
    {"i32",read_i32,METH_O,"Read an int32 at a valid engine address."},
    {"u64",read_u64,METH_O,"Read a uint64 at a valid engine address."},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,"_native_features",NULL,-1,methods};
PyMODINIT_FUNC PyInit__native_features(void) {
    for (int i=0;i<KEY_COUNT;++i) { keys[i]=PyUnicode_InternFromString(names[i]); if (!keys[i]) return NULL; }
    return PyModule_Create(&module);
}
