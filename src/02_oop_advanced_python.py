# %% [markdown]
# # 02 · OOP & Advanced Python — Complete Function Reference
#
# Notebook 02 takes the functions and data structures from `01` and levels
# them up: **classes and the four pillars**, file handling, serialisation,
# exception handling, modules, namespaces, **decorators**, recursion,
# **iterators** and **generators**. Same format as before — every concept is a
# `What / Syntax / Parameters / Returns / Example` block followed by a runnable
# code cell, ending with a quick revision sheet.
#
# | You will master | Section |
# |---|---|
# | Classes, encapsulation, class/static methods | 1–3 |
# | Inheritance, polymorphism, abstraction | 4–6 |
# | Project-style OOP example | 7 |
# | Files & serialisation (pickle/json/csv) | 8–9 |
# | Exceptions, modules, namespaces | 10–12 |
# | Decorators, recursion, iterators, generators | 13–16 |
#
# > **Note:** everything below runs top-to-bottom in a fresh kernel
# > (Python 3.10, standard library only, no internet). File examples write
# > into `tempfile` and clean up after themselves.

# %% [markdown]
# ## ▶ Sessions covered
#
# | # | Session | Video |
# |---|---------|-------|
# | 17 | OOP Part 1 \| Class & Object | https://www.youtube.com/watch?v=Trvqq5w7-0Q |
# | 19 | OOP Part 2 \| Encapsulation & Static Keyword | https://www.youtube.com/watch?v=P4xizq3CiJg |
# | 21 | OOP Part 3 \| Inheritance & Polymorphism | https://www.youtube.com/watch?v=bEWwM4hXZg8 |
# | 22 | OOP Project | https://www.youtube.com/watch?v=awjzebtHQ_o |
# | 26 | What is Abstraction | https://www.youtube.com/watch?v=O-NrQvtorp0 |
# | 27 | Session 10 - File Handling + Serialization | https://www.youtube.com/watch?v=o-TAYRMQzIQ |
# | 29 | Session 11 - Exception Handling & Modules and Packages | https://www.youtube.com/watch?v=rvKR6tciJ2Q |
# | 31 | Session 12 - Decorators & Namespaces | https://www.youtube.com/watch?v=aMARZGTbULc |
# | 33 | Recursion using Python | https://www.youtube.com/watch?v=cNvZK0Wyoik |
# | 34 | What is Iterators in Python? | https://www.youtube.com/watch?v=pH7YVRhnpUI |
# | 35 | Generators in Python | https://www.youtube.com/watch?v=ZfJoU67tG1A |

# %%
import sys

print(sys.version_info[:3], "— Python", sys.version.split()[0])

# %% [markdown]
# ---
# ## 1. Classes & objects
#
# ▶ OOP Part 1 | Class & Object: https://www.youtube.com/watch?v=Trvqq5w7-0Q

# %% [markdown]
# ### `class Name:` + `def __init__(self, ...):`
# **What it does:** `class` defines a blueprint; calling the class creates an
# *instance* (object). `__init__` is the initialiser that runs automatically on
# every creation, and `self` is the instance the method was called on.
# **Syntax:**
# ```python
# class Dog:
#     kind = "canine"                 # class attribute (shared)
#     def __init__(self, name):       # runs at Dog(...)
#         self.name = name            # instance attribute (per object)
# ```
# **Returns:** `__init__` returns `None`; the class call returns the new object.

# %%
class Dog:
    species = "canine"                      # CLASS attribute — shared by all

    def __init__(self, name, age=1):        # instance attributes
        self.name = name                    # self is the object being built
        self.age = age

    def bark(self):                         # instance method — gets self free
        return f"{self.name} says woof!"

rex = Dog("Rex")
print(rex.bark(), "| species:", rex.species)
print("attributes:", rex.name, rex.age)

# %%
# instance vs class attributes — the classic gotcha
class Counter:
    total = 0                     # CLASS attribute

    def __init__(self):
        self.value = 0            # INSTANCE attribute

a, b = Counter(), Counter()
a.value = 10
Counter.total = 99                # shared — visible from both instances
print("a.value:", a.value, "| b.value:", b.value)      # separate objects
print("a.total:", a.total, "| b.total:", b.total)      # same class attribute
print("b.total changed when we edited Counter.total:", b.total == 99)

# NOTE: `a.value = 10` created a NEW instance attribute shadowing nothing,
#       while `a.total = 5` would also shadow — assignments never touch the class.

# %%
# introspection: __dict__, type(), attribute lookup order
rex = Dog("Rex", 4)
print("type(rex)  :", type(rex), "| type(Dog):", type(Dog))
print("rex.__dict__:", rex.__dict__)
print("Dog.__dict__ keys:", [k for k in vars(Dog) if not k.startswith("__")])
print("class name:", Dog.__name__, "| doc:", repr(Dog.__doc__))
print("hasattr:", hasattr(rex, "name"), hasattr(rex, "colour"))
print("getattr with default:", getattr(rex, "colour", "unknown"))
print("bound method:", rex.bark, "| lives on the class:", Dog.bark)

# %% [markdown]
# ### Object creation walkthrough
# `Dog("Rex")` is really `Dog.__new__(Dog)` → fresh object → `__init__(obj, "Rex")`.
# You almost never override `__new__` (it allocates); `__init__` only
# initialises. MRO decides method lookup: `instance → its class → superclasses`.
#
# > **Gotcha — class vs instance variable:** mutating a *mutable* class
# > attribute (a list/dict) leaks between instances; assign per instance in
# > `__init__` instead.

# %%
class Shared:
    tags = []                      # DANGEROUS: one list shared by everyone

x1, x2 = Shared(), Shared()
x1.tags.append("from x1")
print("leaked to the other instance:", x2.tags)

class Safe:
    def __init__(self):
        self.tags = []             # SAFE: fresh list per instance

y1, y2 = Safe(), Safe()
y1.tags.append("from y1")
print("isolated now:", y1.tags, y2.tags)

# %%
# __new__ mention — allocation happens before initialisation
class Meta(type):
    def __call__(cls, *a, **k):
        print("  (type.__call__ -> __new__ then __init__)")
        return super().__call__(*a, **k)

class Thing(metaclass=Meta):
    def __new__(cls, *a, **k):
        print("  __new__ allocates the object")
        return super().__new__(cls)
    def __init__(self):
        print("  __init__ fills it in")

obj = Thing()
print("created:", type(obj).__name__)

# %% [markdown]
# **Section notes — classes:**
#
# | Piece | Meaning |
# |---|---|
# | `class Name:` | blueprint; calling it builds an object |
# | `__init__(self, ...)` | constructor — runs on every `Name(...)` |
# | `self` | the current instance (passed implicitly) |
# | class attribute | shared, defined at class level |
# | instance attribute | private to each object, set in `__init__` |
# | `obj.__dict__` / `vars(obj)` | all instance attributes as a dict |
# | `type(obj)` / `obj.__class__` | the class behind an object |
# | `__new__` vs `__init__` | allocate vs initialise |

# %% [markdown]
# ---
# ## 2. Encapsulation
#
# ▶ OOP Part 2 | Encapsulation & Static Keyword: https://www.youtube.com/watch?v=P4xizq3CiJg

# %%
# _protected (convention) and __private (name mangling)
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner          # public
        self._branch = "Mumbai"     # _protected: "don't touch from outside"
        self.__balance = balance    # __private: name-mangled at class definition

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self.__balance += amount    # internal access is un-mangled for us
        return self.__balance

    def get_balance(self):
        return self.__balance

acc = BankAccount("Ada", 100)
print("public  :", acc.owner)
print("protected (by convention):", acc._branch)
print("deposit ->", acc.deposit(50))

try:
    print(acc.__balance)
except AttributeError as e:
    print("AttributeError:", e)
print("mangled name is accessible as:", acc._BankAccount__balance)
# NOTE: mangling is a safety net against ACCIDENTAL access, not real privacy.

# %% [markdown]
# ### `@property` — getters/setters/deleters that feel like attributes
# **What it does:** turns a method into an attribute-like accessor so you can
# add validation without breaking the calling syntax `obj.attr`.
# **Syntax:**
# ```python
# class C:
#     @property
#     def x(self): ...        # getter  -> obj.x
#     @x.setter
#     def x(self, v): ...     # setter  -> obj.x = v
#     @x.deleter
#     def x(self): ...        # deleter -> del obj.x
# ```

# %%
class Temperature:
    def __init__(self, celsius=0.0):
        self._celsius = celsius      # store privately, expose via property

    @property
    def celsius(self):
        """Getter."""
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        """Setter WITH validation."""
        if not isinstance(value, (int, float)):
            raise TypeError("celsius must be a number")
        if value < -273.15:
            raise ValueError("below absolute zero!")
        self._celsius = value

    @property
    def fahrenheit(self):
        """Computed read-only property — no setter exists."""
        return self._celsius * 9 / 5 + 32

    @celsius.deleter
    def celsius(self):
        self._celsius = 0.0

t = Temperature(100)
print("celsius:", t.celsius, "| fahrenheit:", t.fahrenheit)
t.celsius = 25                       # goes through the setter
print("after set:", t.celsius, t.fahrenheit)
for bad, exc in ((300, ValueError), ("hot", TypeError)):
    try:
        t.celsius = bad
    except Exception as e:
        print(type(e).__name__, "->", e)

# %%
# properties vs get_x()/set_x() methods
class OldStyle:
    def __init__(self, v):
        self.__v = v
    def get_v(self):                 # Java-style: noisy call sites
        return self.__v
    def set_v(self, v):
        self.__v = v

class NewStyle:
    def __init__(self, v):
        self._v = v
    @property
    def v(self):                     # stays an attribute at the call site
        return self._v
    @v.setter
    def v(self, value):
        self._v = value

o, n = OldStyle(1), NewStyle(1)
print("old call style:", o.get_v(), "-> new call style:", n.v)
n.v = 5
print("assignment now validates later too:", n.v)
# TIP: start with a plain attribute; promote to @property when you need
#      validation or computed values — the calling code never has to change.

# %% [markdown]
# **Section notes — encapsulation:**
#
# | Convention | Syntax | Reality |
# |---|---|---|
# | public | `self.x` | open to all |
# | protected | `self._x` | "internal" by convention only |
# | private | `self.__x` | mangled to `self._Class__x` |
# | property | `@property` / `@x.setter` / `@x.deleter` | attribute syntax + logic |
#
# > **Gotcha:** `obj.__x` from outside raises `AttributeError`, but
# > `obj._Class__x` works — mangling prevents *accidental* clashes, not
# > determined access.

# %% [markdown]
# ---
# ## 3. `classmethod` & `staticmethod`
#
# ▶ OOP Part 2 | Encapsulation & Static Keyword: https://www.youtube.com/watch?v=P4xizq3CiJg

# %%
from datetime import datetime

class Employee:
    raise_pct = 0.10                     # class-level state

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def applied(self):                   # instance method — needs an instance
        return round(self.salary * (1 + self.raise_pct), 2)

    @classmethod
    def from_record(cls, record):        # FACTORY: cls is the class itself
        name, salary = record.split(":")
        return cls(name, int(salary))

    @classmethod
    def set_raise_pct(cls, pct):         # mutates CLASS-level state
        cls.raise_pct = pct

    @staticmethod
    def is_workday(day):                 # no cls, no self — pure utility
        return day.weekday() < 5

e = Employee.from_record("Ada:50000")    # alternative constructor
print("factory built:", e.name, e.salary, "| applied:", e.applied())

Employee.set_raise_pct(0.20)
print("after class-level change:", e.applied())     # all instances see it
print("staticmethod:", Employee.is_workday(datetime(2026, 10, 7))),      # Wednesday
print("also callable on an instance:", e.is_workday(datetime(2026, 10, 10)))

# %%
# class-level state: count instances
class Session:
    created = 0
    def __init__(self):
        Session.created += 1            # or type(self).created += 1
    @classmethod
    def how_many(cls):
        return cls.created

for _ in range(4):
    Session()
print("instances made:", Session.how_many())

# %%
# more factory patterns: datetime-style and named constructors
class Date:
    def __init__(self, year, month, day):
        self.year, self.month, self.day = year, month, day

    def __repr__(self):
        return f"Date({self.year}, {self.month}, {self.day})"

    @classmethod
    def from_string(cls, s):            # "2026-10-07"
        y, m, d = map(int, s.split("-"))
        return cls(y, m, d)

    @classmethod
    def today(cls):                     # delegate to the stdlib
        now = datetime.now()
        return cls(now.year, now.month, now.day)

print(Date.from_string("2026-10-07"))
print("today-ish:", Date.today().year > 2020)
print("classmethod goes through MRO:", type(Date.from_string("2020-01-01")).__name__)

# %% [markdown]
# **Section notes — class/static methods:**
#
# | Decorator | Receives | Use for |
# |---|---|---|
# | plain method | `self` (instance) | per-object behaviour |
# | `@classmethod` | `cls` (the class) | factories, class config, counters |
# | `@staticmethod` | neither | utilities that belong to the class namespace |
#
# > **TIP:** `@classmethod` as an alternative constructor (`from_x`,
# > `from_dict`, `from_json`) is the convention behind `dict()`,
# > `datetime.fromisoformat()`, `Decimal.from_float()`.

# %% [markdown]
# ---
# ## 4. Inheritance
#
# ▶ OOP Part 3 | Inheritance & Polymorphism: https://www.youtube.com/watch?v=bEWwM4hXZg8

# %%
# single inheritance
class Animal:
    def __init__(self, name):
        self.name = name
    def speak(self):
        return "..."

class Dog(Animal):                        # Dog inherits from Animal
    def speak(self):                      # method OVERRIDE
        return f"{self.name} says woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says meow!"

for pet in (Dog("Rex"), Cat("Misty")):
    print(pet.speak(), "| inherited attr:", pet.name)

# %%
# multiple & multilevel & hierarchical inheritance
class Engine:
    def start(self):
        return "engine on"

class Wheels:
    def roll(self):
        return "wheels rolling"

class Car(Engine, Wheels):                # MULTIPLE: two parents
    pass

class SportsCar(Car):                     # MULTILEVEL: Car -> Engine/Wheels
    def __repr__(self):
        return "SportsCar"

class Truck(Engine):                      # HIERARCHICAL: Engine has many children
    pass

c = Car()
print("multiple      :", c.start(), "|", c.roll())
print("multilevel    :", type(SportsCar()).__mro__[1].__name__)
print("hierarchical  :", isinstance(Truck(), Engine))

# %%
# MRO — method resolution order
class A:
    def who(self): return "A"
class B(A):
    def who(self): return "B"
class C(A):
    def who(self): return "C"
class D(B, C):
    def who(self): return "D"

print("D.__mro__  :", [cls.__name__ for cls in D.__mro__])
print("D.who ->", D().who(), "| via B then C then A (C3 linearisation)")
# every class appears exactly once; a parent NEVER precedes its child

# %%
# super() — cooperative initialisation up the chain
class Vehicle:
    def __init__(self, make):
        self.make = make
        print("  Vehicle.__init__:", make)

class Electric(Vehicle):
    def __init__(self, make, battery_kwh):
        super().__init__(make)            # run the parent's __init__ first
        self.battery_kwh = battery_kwh
        print("  Electric.__init__:", battery_kwh)

class Tesla(Electric):
    def __init__(self, model, battery_kwh=75):
        super().__init__(model, battery_kwh)   # next class in the MRO
        print("  Tesla.__init__:", model)

t = Tesla("Model 3")
print("state:", t.make, t.battery_kwh)

# %%
# isinstance / issubclass
print("isinstance:", isinstance(t, Tesla), isinstance(t, Electric),
      isinstance(t, Vehicle), isinstance(t, object))
print("not       :", isinstance(t, dict))
print("issubclass:", issubclass(Tesla, Electric), issubclass(Tesla, Vehicle),
      issubclass(Tesla, object), issubclass(Vehicle, Tesla))
print("quick type check:", type(t) is Tesla)

# %% [markdown]
# > **Gotcha:** `isinstance(x, Animal)` is True for every subclass — good for
# > interface checks, dangerous in `if/elif` chains (replace them with
# > polymorphism, section 5).
#
# **Section notes — inheritance:**
#
# | Form | Syntax |
# |---|---|
# | single | `class Dog(Animal):` |
# | multiple | `class Car(Engine, Wheels):` |
# | multilevel | `class SportsCar(Car):` |
# | hierarchical | many children of one parent |
# | inspect lookup | `print(C.__mro__)` / `type.mro(C)` |
# | parent init | `super().__init__(...)` |
# | checks | `isinstance(obj, C)`, `issubclass(C, Base)` |

# %% [markdown]
# ---
# ## 5. Polymorphism
#
# ▶ OOP Part 3 | Inheritance & Polymorphism: https://www.youtube.com/watch?v=bEWwM4hXZg8

# %%
# 1) overriding — same message, different implementations
def make_it_speak(animal):
    print(animal.speak())                 # works for ANY object with .speak()

make_it_speak(Dog("Rex"))
make_it_speak(Cat("Misty"))

# 2) DUCK TYPING — "if it quacks like a duck, it is a duck"
class Robot:
    def speak(self):
        return "beep boop"

make_it_speak(Robot())                    # no inheritance involved at all!

# %%
# 3) operator overloading via dunder methods
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):                    # str(obj) — friendly display
        return f"Vector({self.x}, {self.y})"

    def __repr__(self):                   # repr(obj) — unambiguous, for developers
        return f"Vector({self.x!r}, {self.y!r})"

    def __len__(self):                    # len(obj)
        return 2

    def __eq__(self, other):              # ==  (return NotImplemented for others)
        if not isinstance(other, Vector):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):              # <  (sorts by magnitude)
        return self._mag() < other._mag()

    def __add__(self, other):             # +
        return Vector(self.x + other.x, self.y + other.y)

    def __getitem__(self, i):             # obj[i]
        return (self.x, self.y)[i]

    def __call__(self):                   # obj() — make instances callable
        return self._mag()

    def _mag(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

v1, v2 = Vector(3, 4), Vector(1, 1)
print("str     :", str(v1))
print("repr    :", repr(v1))
print("len     :", len(v1))
print("eq      :", v1 == Vector(3, 4), "|", v1 == v2)
print("lt      :", v1 < v2, "| sorted:", sorted([v1, v2]))
print("add     :", v1 + v2)
print("getitem :", v1[0], v1[1])
print("call    :", v1(), "-> magnitude")
print("f-string uses __str__:", f"{v1}")

# %%
# comparison operators cheat sheet (return NotImplemented to defer/raise)
class Money:
    def __init__(self, amount):
        self.amount = amount
    def __eq__(self, other):
        if isinstance(other, Money):
            return self.amount == other.amount
        return NotImplemented
    def __lt__(self, other):
        if isinstance(other, Money):
            return self.amount < other.amount
        return NotImplemented
    def __hash__(self):
        return hash(self.amount)          # defining __eq__ kills the default hash!
    def __repr__(self):
        return f"Money({self.amount})"

m1, m2 = Money(10), Money(20)
print("Money eq/lt:", m1 == m2, m1 < m2, m1 != m2, sorted([m2, m1]))
print("hashable again:", len({Money(1), Money(1), Money(2)}))
try:
    Money(1) == "10"
except TypeError as e:
    print("TypeError:", e)
# NOTE: == returns NotImplemented -> Python falls back -> TypeError for Money vs str

# %% [markdown]
# **Section notes — polymorphism:**
#
# | Mechanism | Example |
# |---|---|
# | overriding | subclass redefines `speak()` |
# | duck typing | any object with `.speak()` works |
# | display | `__str__` (users), `__repr__` (developers) |
# | size | `__len__` |
# | compare | `__eq__`, `__lt__`, `__gt__`, `__le__`, `__ge__`, `__ne__` |
# | arithmetic | `__add__`, `__sub__`, `__mul__`, `__truediv__` |
# | indexing | `__getitem__`, `__setitem__`, `__contains__` |
# | callable | `__call__` |
#
# > **Gotcha:** defining `__eq__` removes the inherited `__hash__` unless you
# > also define `__hash__` — otherwise instances can no longer be set members
# > or dict keys.

# %% [markdown]
# ---
# ## 6. Abstraction
#
# ▶ What is Abstraction: https://www.youtube.com/watch?v=O-NrQvtorp0

# %%
# ABC + @abstractmethod — define the INTERFACE, hide the implementation
from abc import ABC, abstractmethod

class Shape(ABC):                          # abstract class: cannot be instantiated
    @abstractmethod
    def area(self):
        """Every concrete shape MUST implement this."""

    @abstractmethod
    def perimeter(self):
        ...

    def describe(self):                    # concrete methods are inherited as-is
        return f"{type(self).__name__}: area={self.area():.2f}"

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        return 3.14159 * self.radius ** 2
    def perimeter(self):
        return 2 * 3.14159 * self.radius

class Rectangle(Shape):
    def __init__(self, w, h):
        self.w, self.h = w, h
    def area(self):
        return self.w * self.h
    def perimeter(self):
        return 2 * (self.w + self.h)

for s in (Circle(2), Rectangle(3, 4)):
    print(s.describe(), "| perimeter:", round(s.perimeter(), 2))

try:
    Shape()                                # abstract -> TypeError
except TypeError as e:
    print("TypeError:", e)
print("abstract methods still missing:",
      Shape.__abstractmethods__)

# %%
# composition vs inheritance — "has-a" often beats "is-a"
class Engine2:
    def __init__(self, hp):
        self.hp = hp
    def run(self):
        return f"{self.hp}hp engine running"

class CarWith:                             # COMPOSITION: car HAS an engine
    def __init__(self, hp):
        self.engine = Engine2(hp)         # delegate construction
    def drive(self):
        return self.engine.run() + " -> wheels spin"

class Sedan(CarWith):                      # INHERITANCE: sedan IS a car
    def drive(self):
        return "quiet: " + super().drive()

print("composition:", CarWith(150).drive())
print("inheritance:", Sedan(180).drive())
# TIP: prefer composition when you need to swap parts at runtime
#      (test doubles, multiple engine types, avoids deep hierarchies).

# %% [markdown]
# **Section notes — abstraction:**
#
# | Concept | Code |
# |---|---|
# | abstract base class | `class C(ABC):` |
# | abstract method | `@abstractmethod def m(self): ...` |
# | block instantiation | subclass must implement every abstract method |
# | interface example | `Shape` defines `area`/`perimeter`, subclasses implement |
# | composition | `self.engine = Engine(...)` — delegate, don't inherit |
#
# > **Gotcha:** `@abstractmethod` only stops *instantiation* of the abstract
# > class — a subclass that skips it is still abstract itself.

# %% [markdown]
# ---
# ## 7. OOP project-style mini example
#
# ▶ OOP Project: https://www.youtube.com/watch?v=awjzebtHQ_o
#
# Everything so far in one system: encapsulation (private state + properties),
# inheritance, polymorphism, `classmethod` factories, abstraction and a little
# `__repr__` hygiene.

# %%
class BankAccount:
    """Base class: private balance, validated property, transaction log."""

    def __init__(self, owner, balance=0):
        self.owner = owner                 # public
        self._opening_branch = "main"      # protected (convention)
        self.__balance = balance           # private (name-mangled)
        self._log = []

    # ---- encapsulated state -------------------------------------------
    @property
    def balance(self):
        return self.__balance

    @balance.setter
    def balance(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("balance must be numeric")
        if value < 0:
            raise ValueError("balance cannot be negative")
        self.__balance = value

    # ---- behaviour -----------------------------------------------------
    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self.__balance += amount
        self._log.append(("deposit", amount))
        return self.__balance

    def withdraw(self, amount):
        if amount > self.__balance:
            raise ValueError("insufficient funds")
        self.__balance -= amount
        self._log.append(("withdraw", amount))
        return self.__balance

    def __repr__(self):
        return f"{type(self).__name__}({self.owner!r}, {self.balance!r})"


class SavingsAccount(BankAccount):
    """Child class: adds interest via a class-level rate."""

    interest_rate = 0.04                    # class attribute

    @classmethod
    def set_rate(cls, rate):                # class-level configuration
        cls.interest_rate = rate

    def apply_interest(self):
        interest = self.balance * self.interest_rate
        self.deposit(interest)
        self._log.append(("interest", interest))
        return self.balance


class StudentAccount(BankAccount):
    """Second child: different rules — every withdrawal pays a 1% fee."""

    fee_rate = 0.01

    def withdraw(self, amount):             # OVERRIDE with new policy
        charged = round(amount * (1 + self.fee_rate), 2)
        if charged > self.balance:
            raise ValueError(f"insufficient funds (need {charged} incl. fee)")
        self.balance = self.balance - charged   # through the property setter
        self._log.append(("withdraw", charged))
        return self.balance


# ---- demo -------------------------------------------------------------
acc = SavingsAccount("Ada", 1000)
acc.deposit(500)
print("after deposit :", acc.balance)
print("after interest:", round(acc.apply_interest(), 2))

try:
    acc.balance = -1                        # property setter blocks it
except ValueError as e:
    print("validation   :", e)

try:
    acc.withdraw(10_000)
except ValueError as e:
    print("guard        :", e)

s = StudentAccount("Ben", 200)
print("Ben withdraws 100 (1% fee) ->", s.withdraw(100), "| repr:", repr(s))
try:
    s.withdraw(1000)                            # different rules than the parent
except ValueError as e:
    print("guard        :", e)
SavingsAccount.set_rate(0.05)
print("rate now:", SavingsAccount.interest_rate, "| transactions:", acc._log)

# %%
# polymorphism in action — one loop, many account types
def payroll(accounts, amount):
    for account in accounts:                # each decides its own withdraw()
        try:
            print(f"  {account.owner:<6} -> takes {amount}: {account.withdraw(amount)}")
        except ValueError as e:
            print(f"  {account.owner:<6} -> refused ({e})")

payroll([SavingsAccount("Ada", 500),
         StudentAccount("Ben", 20),
         BankAccount("Cid", 0)], 30)

# %% [markdown]
# **Section notes — the project shape:**
#
# | Layer | Where it shows up |
# |---|---|
# | encapsulation | `__balance` + `balance` property validation |
# | inheritance | `SavingsAccount(BankAccount)` |
# | polymorphism | `payroll()` calls `withdraw()` on any subtype |
# | class-level config | `interest_rate`, `set_rate()` |
# | `__repr__` | readable debugging output |
# | LSP sanity | every subtype still honours `withdraw()` semantics |

# %% [markdown]
# ---
# ## 8. File handling
#
# ▶ Session 10 - File Handling + Serialization: https://www.youtube.com/watch?v=o-TAYRMQzIQ

# %% [markdown]
# ### `open(file, mode='r', encoding=None, ...)`
# **What it does:** opens a file and returns a file object (a buffered stream).
# **Parameters:**
# - `file` — path (`str` or `pathlib.Path`)
# - `mode` — combination of: `r` read (default) · `w` write (truncates!) ·
#   `a` append · `r+` read-write · `x` create exclusively (fails if exists) ·
#   add `b` for bytes (`rb`/`wb`) · add `t` for text (default)
# - `encoding` — text encoding; always pass `encoding="utf-8"` in real code
# **Returns:** a file handle — **always pair it with `with`** so it closes.

# %%
import os, tempfile, shutil

workdir = tempfile.mkdtemp(prefix="files_")
path = os.path.join(workdir, "notes.txt")

# w = write, TRUNCATES an existing file
with open(path, "w", encoding="utf-8") as f:
    f.write("line one\n")
    f.write("line two\n")

# r = read the whole file (mode defaults to 'r' = text read)
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
print("read():", repr(content))

with open(path) as f:
    print("readline():", repr(f.readline()), end="")   # keeps the newline
    print("readline():", repr(f.readline()), end="")
    print()
with open(path) as f:
    print("readlines():", f.readlines())               # list of lines

with open(path, "a") as f:                # a = append, never truncates
    f.write("line three\n")
with open(path) as f:
    print("after append:", f.read().splitlines())

# %%
# x = create exclusively — fails if the file already exists
try:
    open(path, "x").close()
except FileExistsError as e:
    print("FileExistsError:", os.path.basename(e.filename), "already there")

fresh = os.path.join(workdir, "fresh.txt")
with open(fresh, "x") as f:
    f.write("created by mode 'x'")
print("created fresh file:", os.path.exists(fresh))

# w+ / r+ : read AND write
rw = os.path.join(workdir, "rw.txt")
with open(rw, "w") as f:
    f.write("abcdefghij")
with open(rw, "r+") as f:                 # read-write handle
    f.write("HELLO")                      # overwrite the first 5 bytes
    print("tell() after write:", f.tell())
    f.seek(0)                             # jump back to the start
    print("seek(0) then read:", f.read())

# b = binary mode: bytes in, bytes out (no encoding, no newline translation)
binpath = os.path.join(workdir, "blob.bin")
with open(binpath, "wb") as f:
    f.write(b"\x00\x01\x02binary")
with open(binpath, "rb") as f:
    raw = f.read()
print("binary:", raw, "| type:", type(raw).__name__)

with open(binpath, "rb") as f:
    f.seek(0, os.SEEK_END)                # whence=2 -> end of file
    print("size via seek(0, SEEK_END)/tell():", f.tell())
    f.seek(0, os.SEEK_SET)                # whence=0 -> start
    print("first 5 bytes:", f.read(5))

# %%
# write / writelines — writelines does NOT add newlines for you
lines_path = os.path.join(workdir, "lines.txt")
with open(lines_path, "w") as f:
    f.writelines(["alpha\n", "beta\n", "gamma\n"])
    f.write("delta\n")
with open(lines_path) as f:
    print("writelines:", f.read().splitlines())

# reading line by line WITHOUT loading everything (streams, big files)
with open(path) as f:
    for i, line in enumerate(f, 1):        # a file object is itself iterable
        print(f"  stream line {i}: {line.strip()}")

# %% [markdown]
# > **Gotcha:** `open(path, "w")` **destroys** the previous contents the
# > moment it opens. Use `r+`/`a` when you must keep them, and *always* use
# > `with` — it closes the file even if the body raises.
#
# ### `contextlib.contextmanager` — build your own `with` blocks
# **What it does:** turns a generator with one `yield` into a context manager
# (setup before `yield`, teardown in `finally`).

# %%
from contextlib import contextmanager

@contextmanager
def temp_text_file(text=""):
    """Create a temp file for the block, delete it afterwards."""
    handle = os.path.join(tempfile.gettempdir(), f"ctx_{os.getpid()}.txt")
    with open(handle, "w") as f:           # __enter__-side setup
        f.write(text)
    try:
        yield handle
    finally:
        os.remove(handle)                 # __exit__-side teardown

with temp_text_file("hello context manager") as p:
    with open(p) as f:
        print("inside with:", f.read())
print("cleaned up:", not os.path.exists(p))

# %%
# pathlib.Path — the modern, object-oriented way to handle paths
from pathlib import Path

base = Path(workdir)
target = base / "pathlib.txt"              # / builds paths portably
target.write_text("pathlib rocks")         # read_text/write_text helpers
print("exists:", target.exists(), "| name:", target.name,
      "| stem:", target.stem, "| suffix:", target.suffix)
print("read  :", target.read_text())
print("glob  :", sorted(p.name for p in base.glob("*.txt")))
print("suffix checks:", target.suffix in (".txt", ".md"), "| is_file:", target.is_file())
print("with parents:", target.relative_to(base))

shutil.rmtree(workdir)                     # clean up everything we created
print("temp dir removed:", not os.path.exists(workdir))

# %% [markdown]
# **Section notes — files:**
#
# | Mode | Meaning |
# |---|---|
# | `r` / `rt` | read text (default) — errors if missing |
# | `w` / `wt` | write text — **truncates** |
# | `a` / `at` | append text |
# | `r+` | read + write, position at start |
# | `x` | create new file, fail if it exists |
# | `b` | binary (`rb`, `wb`) — bytes, no encoding |
# | `with open(...)` | auto close (also on exceptions) |
# | `f.seek(off, whence)` / `f.tell()` | 0=SET, 1=CUR, 2=END |
# | modern alternative | `pathlib.Path.read_text()/write_text()` |

# %% [markdown]
# ---
# ## 9. Serialization
#
# ▶ Session 10 - File Handling + Serialization: https://www.youtube.com/watch?v=o-TAYRMQzIQ

# %%
import pickle, json, csv
from datetime import date

sdir = tempfile.mkdtemp(prefix="serial_")

# ---- pickle: arbitrary Python objects -> bytes --------------------------
payload = {"users": [{"id": 1, "tags": (1, 2)}], "meta": {"ok": True}}
pk = os.path.join(sdir, "data.pkl")

with open(pk, "wb") as f:                       # always binary for pickle
    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(pk, "rb") as f:
    restored = pickle.load(f)
print("dump/load round-trip equal:", restored == payload)

blob = pickle.dumps(payload)                    # to bytes, no file needed
print("dumps ->", type(blob).__name__, len(blob), "bytes | loads ->",
      pickle.loads(blob) == payload)

class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

print("pickle keeps types:", pickle.loads(pickle.dumps(Point(3, 4))))
# WARNING: pickle.load executes whatever the payload says — NEVER load a
#          pickle you did not create (see the "unsafe load" note below).

# %%
# ---- json: portable text format (dicts/lists/str/num/bool/None) ---------
js = os.path.join(sdir, "data.json")
obj = {"name": "Ada", "scores": [1, 2, 3], "active": True, "extra": None}

with open(js, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, sort_keys=True)      # indent + sorted keys
with open(js, encoding="utf-8") as f:
    print(f.read())
print("compact dumps:", json.dumps(obj, separators=(",", ":")))
print("round-trip:", json.loads(json.dumps(obj)) == obj)

try:
    json.dumps({"when": date(2026, 10, 7)})
except TypeError as e:
    print("TypeError for non-JSON type:", e)
print("default=str fix:", json.dumps({"when": date(2026, 10, 7)}, default=str))

# %%
# ---- csv: rows in, rows out --------------------------------------------
csv_path = os.path.join(sdir, "scores.csv")
rows = [{"name": "Ada", "score": 95}, {"name": "Bob", "score": 88}]

with open(csv_path, "w", newline="") as f:    # newline="" avoids blank lines
    writer = csv.DictWriter(f, fieldnames=["name", "score"])
    writer.writeheader()
    writer.writerows(rows)

with open(csv_path, newline="") as f:
    for row in csv.DictReader(f):             # dicts keyed by the header
        print("DictReader:", row)

with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "score"], extrasaction="ignore")
    writer.writerow({"name": "Cid", "score": 77, "role": "ignored"})   # extra key dropped
with open(csv_path, newline="") as f:
    print("reader rows:", list(csv.reader(f)))

shutil.rmtree(sdir)
print("cleanup:", not os.path.exists(sdir))

# %% [markdown]
# **Section notes — serialization:**
#
# | | `pickle` | `json` |
# |---|---|---|
# | portability | Python-only | any language |
# | format | binary bytes | human-readable text |
# | types | almost any object | dict/list/str/number/bool/null |
# | custom types | works (same codebase) | needs `default=` hook |
# | security | **never** load untrusted data | safer (no code execution) |
# | typical use | caches, checkpoints, multiprocessing | APIs, config, logs |
#
# > **Gotcha:** `pickle.load()` will run arbitrary code embedded in the file
# > (via `__reduce__`). Treat an untrusted `.pkl` like an untrusted executable.
# > For cross-language interchange, prefer JSON — or `json` + explicit schema.

# %% [markdown]
# ---
# ## 10. Exception handling
#
# ▶ Session 11 - Exception Handling & Modules and Packages: https://www.youtube.com/watch?v=rvKR6tciJ2Q

# %% [markdown]
# ### `try / except / else / finally`
# **What it does:** run risky code, react to specific failures, guarantee
# cleanup.
# **Flow:** `try` → (`except` on error, skipped otherwise) → `else` when the
# body succeeded → `finally` **always**.
#
# ```python
# try:
#     risky()
# except ValueError as e:      # catch SPECIFIC types
#     handle(e)
# else:
#     no_error()               # ran only if nothing was raised
# finally:
#     cleanup()                # runs no matter what
# ```

# %%
def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:              # most specific type first
        print("  caught ZeroDivisionError")
        result = None
    except TypeError as e:
        print("  TypeError:", e)
        result = None
    else:
        print("  else ran: the body succeeded")   # only on success
    finally:
        print("  finally always runs")
    return result

print("ok  :", divide(10, 2))
print("boom:", divide(10, 0))

# catching several types at once
for value in ("x", None, [1]):
    try:
        int(value)
    except (ValueError, TypeError) as e:
        print("caught:", type(e).__name__, "-", e)

# %%
# raising & re-raising
def validate_age(age):
    if not isinstance(age, int):
        raise TypeError("age must be an int")     # explicit failure
    if age < 0:
        raise ValueError("age cannot be negative")
    return age

print("valid  :", validate_age(30))
for bad in ("30", -1):
    try:
        validate_age(bad)
    except Exception as e:
        print("raised :", type(e).__name__, "-", e)

# bare `raise` inside an except block re-raises the SAME exception
def _checked_age(age):
    try:
        validate_age(age)
    except ValueError:
        print("logging then re-raising...")
        raise                                # traceback is preserved

try:
    _checked_age(-5)
except ValueError as e:
    print("outer handler got:", e)

# %%
# raise ... from — chain exceptions ("the original cause of this failure")
def parse_config(text):
    try:
        return int(text)
    except ValueError as e:
        raise RuntimeError("config file is corrupt") from e

try:
    parse_config("oops")
except RuntimeError as exc:
    print("msg   :", exc)
    print("cause :", type(exc.__cause__).__name__, "-", exc.__cause__)
    print("context preserved in the traceback (see __context__)")

# %%
# custom exception class — give callers something specific to catch
class InsufficientBalanceError(Exception):
    """Raised when a withdrawal exceeds the available balance."""

    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"cannot withdraw {amount} from balance {balance}")

class Account:
    def __init__(self, balance):
        self.balance = balance
    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientBalanceError(self.balance, amount)
        self.balance -= amount
        return self.balance

acc = Account(100)
try:
    acc.withdraw(250)
except InsufficientBalanceError as e:
    print("custom error:", e, "| shortfall:", e.amount - e.balance)
except Exception as e:                       # order matters: specific first
    print("generic:", type(e).__name__)
print("still usable:", acc.withdraw(30))

# %%
# EAFP vs LBYL — two ways to guard the same operation
data = {"a": 1}

# LBYL: Look Before You Leap
if "b" in data:
    value = data["b"]
else:
    value = None
print("LBYL:", value)

# EAFP: Easier to Ask Forgiveness than Permission (Pythonic)
try:
    value = data["b"]
except KeyError:
    value = None
print("EAFP:", value)
# TIP: EAFP wins when the check itself is expensive or racy (TOCTOU), and it
#      keeps the happy path unindented. LBYL is fine for simple, cheap guards.

# %%
# the traceback module — capture the error text programmatically
import traceback

try:
    {}["missing"]
except KeyError as e:
    text = traceback.format_exc()
    print("captured traceback (last line):", text.strip().splitlines()[-1])
    print("frames in traceback:", text.count("File"))
    print("exception type     :", type(e).__name__)

# sys.exc_info() gives the current error details
try:
    int("nope")
except Exception:
    print("exc_info ->", type(sys.exc_info()[1]).__name__, sys.exc_info()[1])

# %% [markdown]
# ### Exception hierarchy (partial)
#
# ```text
# BaseException
#  ├── SystemExit  KeyboardInterrupt  GeneratorExit
#  └── Exception
#       ├── TypeError      ValueError   (incl. RegExpError)
#       │    └── UnicodeError (UnicodeDecodeError, UnicodeEncodeError)
#       ├── KeyError       IndexError    AttributeError
#       ├── ZeroDivisionError            ArithmeticError
#       ├── IOError/OSError (FileNotFoundError, PermissionError)
#       ├── StopIteration   StopAsyncIteration
#       ├── RuntimeError    NotImplementedError
#       ├── ImportError     ModuleNotFoundError
#       ├── AssertionError  MemoryError  OverflowError
#       └── Warning (DeprecationWarning, UserWarning ...)
# ```
#
# **Interview Q table**
#
# | Question | Answer |
# |---|---|
# | difference between `try/except` and `if` check? | EAFP vs LBYL — exceptions cost only when raised |
# | what does `finally` guarantee? | it runs on success, error, `return`, `break` — even `SystemExit` |
# | what happens to exceptions in `else`? | `else` runs only when `try` did **not** raise |
# | `raise` vs `raise X from e`? | `from` sets `__cause__` — hides the misleading "during handling" chain |
# | where to catch broadly? | as close to the failure as you can *handle* it; never `except:` bare |
# | can you catch `KeyboardInterrupt`? | technically yes, please don't (Ctrl-C must work) |
# | custom exception? | subclass `Exception`, store structured fields, keep the message useful |
# | what is `except Exception` vs bare? | catches almost everything but NOT `SystemExit`/`KeyboardInterrupt` |

# %% [markdown]
# **Section notes — exceptions:**
#
# | Piece | Role |
# |---|---|
# | `try` | risky block |
# | `except T as e` | handle a specific type |
# | `else` | ran only if no exception |
# | `finally` | always runs (cleanup) |
# | `raise` / `raise X from e` | create / chain exceptions |
# | `class MyError(Exception)` | domain-specific failures |
# | EAFP vs `if` | Pythonic guard style |

# %% [markdown]
# ---
# ## 11. Modules & packages
#
# ▶ Session 11 - Exception Handling & Modules and Packages: https://www.youtube.com/watch?v=rvKR6tciJ2Q

# %%
# three import styles
import math                               # the whole module
from math import sqrt, pi as PI           # selected names, with an alias
import math as m                          # alias the module itself

print(math.floor(2.7), math.gcd(12, 18))
print(sqrt(16), PI, m.ceil(2.1))
from math import factorial, isclose
print(factorial(5), isclose(0.1 + 0.2, 0.3))

# %%
# __name__ and the "__main__" guard
print("__name__ in a notebook/script top level:", __name__)

# this is how every reusable module protects its demo code:
#
#     def helper():
#         ...
#     if __name__ == "__main__":     # True only when run directly
#         helper()
#
# when imported, __name__ becomes the module name instead — so the demo
# code below never runs on import.

# %%
# a real import: write a tiny module, put it on sys.path, import it
import os, tempfile, shutil, sys

mod_dir = tempfile.mkdtemp(prefix="mods_")
mod_path = os.path.join(mod_dir, "greet_mod.py")
with open(mod_path, "w") as f:
    f.write('print("module body runs on import")\n'
            'def hello():\n'
            '    return "hi from greet_mod"\n'
            'if __name__ == "__main__":\n'
            '    print("you ran me as a script")\n')

sys.path.insert(0, mod_dir)               # where Python looks for modules
try:
    import greet_mod
    print("call:", greet_mod.hello(), "| file:", os.path.basename(greet_mod.__file__))
finally:
    sys.path.remove(mod_dir)
    sys.modules.pop("greet_mod", None)    # forget the cached module
    shutil.rmtree(mod_dir)

print("sys.path entries:", len(sys.path), "-> directories Python searches for modules")
print("relative import at top level fails:", end=" ")
try:
    exec("from . import sibling")         # noqa: F821 — no package context
except ImportError as e:
    print(type(e).__name__, "-", e)
# NOTE: relative imports (from . import x) only work INSIDE a package

# %%
# stdlib tour — batteries included (all of these ship with Python)
import os, sys, math, random, datetime, re

print("os      :", os.name, "| cwd exists:", os.path.isdir(os.getcwd()),
      "| HOME set:", "HOME" in os.environ or "USERPROFILE" in os.environ)
print("sys     :", sys.platform, sys.version_info[:2], "| maxsize:", sys.maxsize)
print("math    :", math.sqrt(2), round(math.pi, 4), math.factorial(6),
      math.gcd(12, 18), math.log2(1024))

random.seed(42)                            # TIP: seed for reproducible demos
print("random  :", random.randint(1, 6), random.choice(["a", "b", "c"]),
      round(random.random(), 3), random.sample(range(10), 3))

today = datetime.date(2026, 10, 7)
print("datetime:", today, today.strftime("%A %d %B %Y"),
      "| iso:", datetime.date.fromisoformat("2026-10-07"))
print("timedelta:", datetime.timedelta(days=3, hours=5))

print("re      :", re.findall(r"\d+", "a1 b22 c333"))
print("re.sub  :", re.sub(r"\s+", "-", "too   much   space"))
m = re.search(r"(\w+)@(\w+)", "mail: ada@example here")
print("re.group:", m.group(0), m.groups() if m else None)

# %% [markdown]
# **Section notes — modules:**
#
# | Tool | Use for |
# |---|---|
# | `os` / `pathlib` | paths, environment, filesystem |
# | `sys` | interpreter details: `argv`, `path`, `version`, `exit` |
# | `math` / `statistics` | numeric helpers |
# | `random` / `secrets` | RNG (seeded) / cryptographic random |
# | `datetime` / `time` | dates, durations, clocks |
# | `re` | regular expressions |
# | `json` / `csv` / `pickle` | serialization (section 9) |
# | `collections` / `itertools` / `functools` | data structures, iteration, function tools |
# | `pip install X` | third-party packages (needs network — not here) |
#
# > **Gotcha:** `import x` binds the *module* name; `from x import y` binds
# > the *object* — later changes to `x.y` are **not** seen by `from` importers.
# > Also: `from x import *` pollutes the namespace; never use it outside
# > `__init__.py` or notebooks.

# %% [markdown]
# ---
# ## 12. Namespaces & scope
#
# ▶ Session 12 - Decorators & Namespaces: https://www.youtube.com/watch?v=aMARZGTbULc

# %%
# a namespace is just a mapping of names -> objects
module_ns = globals()
print("globals() is a dict:", isinstance(module_ns, dict),
      "| names in it:", len(module_ns))

def sample():
    local_var = "I am local"
    print("locals() keys   :", list(sample.__code__.co_varnames))
    print("local_var value :", local_var)
    return local_var

print(sample())
print("LEGB lookup order: Local -> Enclosing -> Global -> Built-in")

# %%
# global & nonlocal (recap from notebook 01, now with namespaces in mind)
count = 0
def bump():
    global count
    count += 1
bump(); bump()
print("global count:", count)

def enclosing():
    level = 0
    def inner():
        nonlocal level
        level += 1
        return level
    inner(); inner()
    return level

print("nonlocal level:", enclosing())

# %%
# closures — an inner function that remembers the enclosing scope
def make_multiplier(factor):
    """Return a function that multiplies its argument by `factor`."""
    def multiplier(x):
        return x * factor          # `factor` is a FREE variable
    multiplier.__qualname__ = f"multiplier(x{factor})"
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)
print("closure results:", double(21), triple(7))
print("captured cell contents:", double.__closure__[0].cell_contents)
print("closure cells:", [c.cell_contents for c in double.__closure__])

# %%
# late-binding gotcha — the loop variable is read at CALL time
funcs = [lambda: i for i in range(3)]
print("late binding  :", [f() for f in funcs])           # [2, 2, 2]

funcs = [lambda i=i: i for i in range(3)]                # default-arg fix
print("early binding  :", [f() for f in funcs])          # [0, 1, 2]

# late binding is what you WANT when all callbacks should see the final value
real_double = make_multiplier(2)   # (kept here to prove closures outlive scope)
del make_multiplier                # the closure keeps `factor` alive anyway
print("closure still alive:", real_double(5))

# %% [markdown]
# **Section notes — namespaces:**
#
# | Scope | Where | Written with |
# |---|---|---|
# | Local | current function | normal assignment |
# | Enclosing | outer function (closures) | `nonlocal x` |
# | Global | module namespace | `global x` |
# | Built-in | `builtins` module | never |
# | lookup | LEGB, inner first | `x` alone |
#
# > **Gotcha:** a closure captures the **variable**, not its value at
# > definition time — bind with a default argument (`lambda i=i: ...`) when
# > you need the current value.

# %% [markdown]
# ---
# ## 13. Decorators
#
# ▶ Session 12 - Decorators & Namespaces: https://www.youtube.com/watch?v=aMARZGTbULc

# %%
# recap: functions are objects, and a decorator is just fn -> fn
def shout(text):
    """Uppercase a string."""
    return text.upper()

aliased = shout
print("aliased:", aliased("hey"), "| same object:", aliased is shout)

# the pattern behind every decorator: wrap a function inside another one
def loud(func):                          # takes the ORIGINAL function
    def wrapper(*args, **kwargs):        # adds behaviour around it
        return func(*args, **kwargs).upper() + "!!"
    return wrapper                       # returns the NEW function

def say(msg):
    return f"said {msg}"

decorated = loud(say)                    # what @loud does to `say`
print("decorated call:", decorated("hi"))

# %%
# @decorator syntax — applies the wrapper at definition time
from functools import wraps

def trace(func):
    @wraps(func)                         # keeps __name__, __doc__, __module__
    def wrapper(*args, **kwargs):
        print(f"  entering {func.__name__}{args}")
        result = func(*args, **kwargs)
        print(f"  leaving  {func.__name__} -> {result!r}")
        return result
    return wrapper

@trace
def add(a, b):
    """Add two numbers."""
    return a + b

print("result:", add(2, 3))
print("metadata preserved:", add.__name__, "|", add.__doc__)

def no_wrap(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@no_wrap
def plain():
    """doc"""
print("WITHOUT @wraps, __name__ becomes:", plain.__name__,
      "| __doc__:", plain.__doc__)

# %%
# decorators WITH arguments — one extra layer of nesting
def repeat(n=2):
    """Run the decorated function n times, return the last result."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = None
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(n=3)
def greet(name):
    return f"hi {name}"

print("repeated:", greet("Ada"))
print("stacked decorators (bottom-up):", end=" ")

@repeat(n=2)
@trace
def probe(x):
    return x * 10

print(probe(5))

# %%
# practical 1: a TIMER using time.perf_counter
import time

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  [TIMER] {func.__name__} took {elapsed_ms:.3f} ms")
        return result
    return wrapper

@timer
def sum_squares(n):
    return sum(i * i for i in range(n))

print("sum_squares(200_000) =", sum_squares(200_000))

# %%
# practical 2: a DEBUG logger
def debug(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"  [DEBUG] {func.__name__}({', '.join(map(repr, args))})")
        result = func(*args, **kwargs)
        print(f"  [DEBUG]   -> {result!r}")
        return result
    return wrapper

@debug
def median_pair(a, b):
    return (a + b) / 2

median_pair(4, 7)

# %%
# practical 3: @lru_cache — memoise a recursive function in one line
from functools import lru_cache

naive_calls = 0
def fib_naive(n):
    global naive_calls
    naive_calls += 1
    if n < 2:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)

@lru_cache(maxsize=None)
def fib_cached(n):
    if n < 2:
        return n
    return fib_cached(n - 1) + fib_cached(n - 2)

t0 = time.perf_counter()
r1 = fib_naive(28)
t_naive = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
r2 = fib_cached(28)
t_cached = (time.perf_counter() - t0) * 1000

print(f"fib_naive (28)  = {r1}, calls = {naive_calls:7d}, {t_naive:8.1f} ms  O(2^n)")
print(f"fib_cached(28)  = {r2}, calls = {fib_cached.cache_info().currsize:7d}, "
      f"{t_cached:8.3f} ms  O(n)")
print("cache_info:", fib_cached.cache_info())
fib_cached.cache_clear()
print("after cache_clear:", fib_cached.cache_info())
print("cache uses:", fib_cached.__wrapped__.__name__, "under the hood")

# %% [markdown]
# **Section notes — decorators:**
#
# | Piece | Code |
# |---|---|
# | wrapper pattern | `def deco(f): def wrapper(*a, **k): ...; return wrapper` |
# | syntax | `@decorator` == `f = decorator(f)` |
# | keep metadata | `@wraps(f)` from `functools` |
# | with arguments | three layers: `deco(arg) -> decorator -> wrapper` |
# | stacking | applied bottom-up |
# | stdlib decorators | `functools.lru_cache`, `functools.singledispatch`, `classmethod` |
#
# > **Gotcha:** a decorator must **return a callable**, and `@decorator`
# > without parentheses (i.e. `@trace`) means the decorator receives the
# > function directly — with parentheses (`@repeat(n=3)`) you need the extra
# > `decorator` layer.

# %% [markdown]
# ---
# ## 14. Recursion
#
# ▶ Recursion using Python: https://www.youtube.com/watch?v=cNvZK0Wyoik

# %%
# base case + recursive case — both are mandatory
def factorial(n):
    if n <= 1:                      # BASE case: no further call
        return 1
    return n * factorial(n - 1)      # RECURSIVE case: smaller problem

def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

print("factorial(6) =", factorial(6))
print("fib(10)      =", fib(10))

# %%
# call stack: every call adds a frame until the base case unwinds it
import sys
old_limit = sys.getrecursionlimit()      # CPython ships 1000; ipykernel raises it
print("recursion limit:", old_limit)

def depth(n):
    if n == 0:
        return 0                     # deepest frame reached
    return 1 + depth(n - 1)

print("frames used for depth(10):", depth(10))
print("frames used for depth(999):", depth(999))             # just under the limit

def runaway(n):
    return runaway(n + 1)            # no base case!
try:
    runaway(0)
except RecursionError as e:
    print("RecursionError:", type(e).__name__, "-", e)

sys.setrecursionlimit(2000)
print("limit after setrecursionlimit:", sys.getrecursionlimit())
sys.setrecursionlimit(old_limit)         # restore what the kernel gave us
print("restored:", sys.getrecursionlimit() == old_limit)

# %%
# tree recursion — fib(n) calls itself TWICE: exponential work
def fib_tree(n, depth_=0):
    pad = "  " * depth_
    if n < 2:
        return n
    print(f"{pad}fib({n})")
    left = fib_tree(n - 1, depth_ + 1)
    right = fib_tree(n - 2, depth_ + 1)
    return left + right

print("fib(4) =", fib_tree(4))

# %%
# memoisation: cache results so each subproblem is solved once
def fib_memo(n, cache=None):
    if cache is None:
        cache = {}
    if n in cache:
        return cache[n]
    if n < 2:
        return n
    cache[n] = fib_memo(n - 1, cache) + fib_memo(n - 2, cache)
    return cache[n]

from functools import lru_cache

@lru_cache(maxsize=None)
def fib_lru(n):
    return n if n < 2 else fib_lru(n - 1) + fib_lru(n - 2)

print("memo dict :", fib_memo(50))
print("lru_cache :", fib_lru(50))
print("lru_cache size:", fib_lru.cache_info().currsize)

# %%
# recursion -> iteration: the stack is a LIFO structure you can simulate
def factorial_rec(n):
    return 1 if n <= 1 else n * factorial_rec(n - 1)

def factorial_iter(n):
    result = 1
    for i in range(2, n + 1):        # explicit accumulator instead of frames
        result *= i
    return result

print("recursion:", factorial_rec(10), "| iteration:", factorial_iter(10))
# TIP: convert tail recursion by carrying the accumulator in a loop; convert
#      tree recursion with an explicit stack (list used as LIFO / collections.deque)

# %% [markdown]
# **Section notes — recursion:**
#
# | Piece | Rule |
# |---|---|
# | base case | stops the self-calls — missing it = `RecursionError` |
# | recursive case | must move TOWARD the base case |
# | cost | one stack frame per call (CPython ships 1000 — check `sys.getrecursionlimit()`) |
# | tree recursion | `fib` without cache = O(2ⁿ) |
# | fix 1 | memoise: dict cache or `@lru_cache` → O(n) |
# | fix 2 | rewrite as a loop / explicit stack (no frame overhead) |

# %% [markdown]
# ---
# ## 15. Iterators
#
# ▶ What is Iterators in Python?: https://www.youtube.com/watch?v=pH7YVRhnpUI

# %%
# iterable vs iterator
print("list  is iterable:", hasattr([1, 2], "__iter__"))
print("iter(list) gives  :", type(iter([1, 2])).__name__, "- it has __next__:",
      hasattr(iter([1, 2]), "__next__"))
print("range is BOTH     :", hasattr(range(3), "__iter__"), hasattr(range(3), "__getitem__"))
print("str iterator      :", type(iter("hi")).__name__)

# for-loop DESUGARING — what the interpreter actually does
it = iter([10, 20, 30])               # 1) get an iterator (__iter__)
while True:
    try:
        item = next(it)               # 2) ask for the next value (__next__)
    except StopIteration:             # 3) StopIteration -> leave the loop
        break
    print("  desugared:", item, end=" | ")
print()

# %%
# custom iterator class — the full protocol
class Countdown:
    """Iterable AND iterator; __iter__ restarts the count."""

    def __init__(self, start):
        self.start = start
        self.current = start

    def __iter__(self):               # iter(obj) lands here
        self.current = self.start     # restart for a fresh for-loop
        return self

    def __next__(self):               # next(obj) lands here
        if self.current <= 0:
            raise StopIteration       # signals "no more items"
        self.current -= 1
        return self.current + 1

c = Countdown(3)
print("next() calls:", next(c), next(c), next(c))
try:
    next(c)
except StopIteration:
    print("StopIteration raised when exhausted")
print("in a for loop :", list(Countdown(4)))
print("reusable later :", list(c))                    # __iter__ restarted it

# %%
# a generator-style state machine vs a list-based iterator
class Grid:
    """Iterate over an x*y grid without materialising it."""
    def __init__(self, x, y):
        self.x, self.y, self.i = x, y, 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i >= self.x * self.y:
            raise StopIteration
        value = (self.i // self.y, self.i % self.y)
        self.i += 1
        return value

print("grid:", list(Grid(2, 3)))

# %%
# range as an iterator (and a sequence)
r = range(0, 10, 2)
print("range object :", list(r), "| len:", len(r), "| index:", r[1])
ri = iter(r)
print("iter(range)  :", type(ri).__name__, next(ri), next(ri))
print("range reusable:", list(r))            # each for-loop builds a NEW iterator
print("one-shot iterators:", type(zip([1], [2])).__name__,
      type(map(str, [1])).__name__,
      type(filter(None, [1])).__name__)
# NOTE: lists/strings/ranges are re-iterable; zip/map/filter/generators are one-shot

# %% [markdown]
# **Section notes — iterators:**
#
# | Protocol | Method | Contract |
# |---|---|---|
# | iterable | `__iter__(self)` | returns an iterator |
# | iterator | `__next__(self)` | returns next value or raises `StopIteration` |
# | builtins | `iter(x)`, `next(it)` | call those dunders for you |
# | loop sugar | `for` | `iter()` + `next()` + catch `StopIteration` |
# | re-iterable | list, str, range, tuple | new iterator per loop |
# | one-shot | zip, map, filter, generators | exhausted after one pass |
#
# > **Gotcha:** returning `self` from `__iter__` makes the object
# > single-use-in-place — to support nested/repeated loops, return a **fresh
# > iterator** (as `range` and generators do).

# %% [markdown]
# ---
# ## 16. Generators
#
# ▶ Generators in Python: https://www.youtube.com/watch?v=ZfJoU67tG1A
#
# A generator is the shortcut to writing an iterator: any function with
# `yield` returns a **generator object** that produces values one at a time,
# remembering its local state between calls.

# %%
def countdown_gen(n):
    print("  (body starts running NOW)")
    while n > 0:
        yield n                   # PAUSES here, sends the value out
        n -= 1
    print("  (generator finished)")

g = countdown_gen(3)              # NOTHING runs yet — just builds the object
print("created:", type(g).__name__, "| is iterator:", hasattr(g, "__next__"))
print("first  :", next(g))        # runs until the first yield
print("rest   :", list(g))        # resumes until StopIteration
try:
    next(g)
except StopIteration:
    print("exhausted generators cannot be rewound")

# %%
# laziness — compute only what the consumer asks for
def trace_numbers(limit):
    for i in range(limit):
        print(f"  producing {i}")
        yield i

g = trace_numbers(3)
print("no output until we pull:")
print("just one:", next(g))       # only ONE value was produced
print("now the rest:", list(g))

# %%
# memory: generator vs list for the SAME 1,000,000 items
import sys
big_list = list(range(1_000_000))
big_gen = (i for i in range(1_000_000))          # generator EXPRESSION

print(f"list overhead (pointers only): {sys.getsizeof(big_list):>12,} bytes")
print(f"plus int objects: ~28 bytes each -> roughly {sys.getsizeof(big_list) + 1_000_000 * 28:,} bytes")
print(f"generator object                : {sys.getsizeof(big_gen):>12,} bytes")
print("the generator stores only code + cursor — O(1) memory")
del big_list, big_gen                           # be a good citizen

# %%
# generator expressions look like list comprehensions, without the brackets
squares_g = (x * x for x in range(5))           # () instead of []
print(type(squares_g).__name__, list(squares_g))
print("sum of a genexpr (no intermediate list):", sum(i * i for i in range(100_000)))
first = next(i for i in range(10, 30) if i % 7 == 0)
print("first multiple of 7 from 10:", first)
print("cannot reuse:", list(squares_g))         # already consumed -> []

# %%
# yield from — delegate to a sub-generator (flattens it)
def inner():
    yield from [1, 2, 3]        # yields 1, 2, 3 one by one
    yield from "ab"             # yields 'a', 'b'

def outer():
    print("  entering outer")
    yield from inner()
    yield 99
    print("  leaving outer")

print("yield from ->", list(outer()))

# %%
# infinite generator + bounded consumption
import itertools

def naturals(start=0):
    n = start
    while True:                 # no base case — this is the POINT
        yield n
        n += 1

print("take 6     :", list(itertools.islice(naturals(), 6)))
print("from 100   :", list(itertools.islice(naturals(100), 3)))

def take(gen, k):               # manual bounded consumption
    out = []
    for _ in range(k):
        out.append(next(gen))   # would loop forever without the bound
    return out

print("manual take:", take(naturals(7), 4))
print("islice is itself a generator:", list(itertools.islice(naturals(), 3)))

# %%
# a pipeline — each stage is lazy, so items stream through in O(1) memory
def read_numbers():
    yield from [3, 8, 15, 16, 23, 42]

def only_evens(nums):
    for n in nums:
        if n % 2 == 0:
            yield n

def squared(nums):
    for n in nums:
        yield n * n

pipeline = squared(only_evens(read_numbers()))   # compose generators
print("pipeline result:", list(pipeline))

# generators are one-shot consumers: chain fresh ones per pass
for _ in range(2):
    print("  pass:", list(squared(only_evens(read_numbers()))))

# %% [markdown]
# | | `list(range(n))` | `(x for x in range(n))` |
# |---|---|---|
# | memory | O(n) — all items in RAM | O(1) — code + cursor |
# | first value | after building everything | instantly, on demand |
# | reuse | yes (`for` twice is fine) | no — one-shot, re-create it |
# | indexing / len | yes | no |
# | best for | small, re-used, random-access data | streams, pipelines, huge/infinite data |
#
# **Section notes — generators:**
#
# | Piece | Code |
# |---|---|
# | define | `def g(): ... yield value` |
# | pull one value | `next(g)` (pauses/resumes the function) |
# | exhaustion | raises `StopIteration` |
# | gen expression | `(expr for x in it if cond)` |
# | delegate | `yield from sub_generator()` |
# | bounded infinite | `itertools.islice(gen, k)` |
# | pipeline | `stage3(stage2(stage1()))` — lazy end to end |
#
# > **Gotcha:** a generator is **one-shot**: after exhaustion it keeps
# > returning `StopIteration` — call the generator *function* again for a new
# > object. And because it is lazy, the body does not run until you pull.

# %% [markdown]
# ---
# ## Quick revision sheet
#
# | Task | Code |
# |---|---|
# | define a class | `class C:` + `def __init__(self, ...)` |
# | private attribute | `self.__x` (mangled `self._C__x`) |
# | getter/setter | `@property` / `@x.setter` |
# | factory constructor | `@classmethod def from_x(cls, ...)` |
# | utility method | `@staticmethod def helper(...)` |
# | inherit | `class Child(Parent):` + `super().__init__(...)` |
# | lookup order | `print(C.__mro__)` |
# | type checks | `isinstance(obj, C)`, `issubclass(C, Base)` |
# | custom display | `__str__` (friendly), `__repr__` (precise) |
# | operators | `__eq__ __lt__ __add__ __len__ __getitem__ __call__` |
# | interface | `class I(ABC): @abstractmethod def m(self)` |
# | open a file safely | `with open(p) as f:` |
# | write structured data | `json.dump(obj, f, indent=2)` / `csv.DictWriter` |
# | pickle rule | `dump`/`load` in **binary**; never load untrusted pickles |
# | catch an error | `try: ... except Specific as e: ... finally: ...` |
# | chain causes | `raise New from original` |
# | guard script code | `if __name__ == "__main__":` |
# | wrap a function | `@decorator` + `functools.wraps` |
# | cache a function | `@lru_cache(maxsize=None)` |
# | recursion fix | memoise (dict / `lru_cache`) or convert to a loop |
# | custom iterator | `__iter__` + `__next__` + `raise StopIteration` |
# | lazy sequence | `yield` inside a generator function |
#
# > **Interview short-list:** `__init__` vs `__new__` · why mangling is not
# > privacy · `classmethod` vs `staticmethod` · MRO/C3 and `super()` ·
# > `__str__` vs `__repr__` · why `__eq__` needs `__hash__` · abstract vs
# > duck typing · EAFP vs LBYL · `pickle` vs `json` · decorator layers ·
# > closure late binding · iterable vs iterator · generator memory O(1).

# %% [markdown]
# ---
# **Next notebook:** `03_numpy.ipynb` — leave the Python interpreter behind
# and start vectorising: arrays, broadcasting, indexing and ufuncs.
#
# **Playlist:** https://www.youtube.com/playlist?list=PLKnIA16_RmvbAlyx4_rdtR66B7EHX5k3z
