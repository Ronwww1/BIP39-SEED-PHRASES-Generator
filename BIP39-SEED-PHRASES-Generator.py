#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенный BIP39 Генератор Seed Фраз с проверкой уникальности
Автор: AI Assistant
Версия: 2.0
"""

import os
import json
import hashlib
import secrets
import qrcode
from datetime import datetime
from typing import List, Optional, Dict, Any
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import io

class BIP39Generator:
    """
    Класс для генерации уникальных BIP39 seed фраз с проверкой повторений
    """
    
    # BIP39 словарь (английский)
    WORDLIST = [
        "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract", 
        "absurd", "abuse", "access", "accident", "account", "accuse", "achieve", "acid",
        "acoustic", "acquire", "across", "act", "action", "actor", "actress", "actual",
        "adapt", "add", "addict", "address", "adjust", "admit", "adult", "advance",
        "advice", "aerobic", "affair", "afford", "afraid", "again", "age", "agent",
        "agree", "ahead", "aim", "air", "airport", "aisle", "alarm", "album",
        "alcohol", "alert", "alien", "all", "alley", "allow", "almost", "alone",
        "alpha", "already", "also", "alter", "always", "amateur", "amazing", "among",
        "amount", "amused", "analyst", "anchor", "ancient", "anger", "angle", "angry",
        "animal", "ankle", "announce", "annual", "another", "answer", "antenna", "antique",
        "anxiety", "any", "apart", "apology", "appear", "apple", "approve", "april",
        "arch", "arctic", "area", "arena", "argue", "arm", "armed", "armor",
        "army", "around", "arrange", "arrest", "arrive", "arrow", "art", "artefact",
        "artist", "artwork", "ask", "aspect", "assault", "asset", "assist", "assume",
        "asthma", "athlete", "atom", "attack", "attend", "attitude", "attract", "auction",
        "audit", "august", "aunt", "author", "auto", "autumn", "average", "avocado",
        "avoid", "awake", "aware", "away", "awesome", "awful", "awkward", "axis",
        "baby", "bachelor", "bacon", "badge", "bag", "balance", "balcony", "ball",
        "bamboo", "banana", "banner", "bar", "barely", "bargain", "barrel", "base",
        "basic", "basket", "battle", "beach", "bean", "beauty", "because", "become",
        "beef", "before", "begin", "behave", "behind", "believe", "below", "belt",
        "bench", "benefit", "best", "betray", "better", "between", "beyond", "bicycle",
        "bid", "bike", "bind", "biology", "bird", "birth", "bitter", "black",
        "blade", "blame", "blanket", "blast", "bleak", "bless", "blind", "blood",
        "blossom", "blouse", "blue", "blur", "blush", "board", "boat", "body",
        "boil", "bomb", "bone", "bonus", "book", "boost", "border", "boring",
        "borrow", "boss", "bottom", "bounce", "box", "boy", "bracket", "brain",
        "brand", "brass", "brave", "bread", "breeze", "brick", "bridge", "brief",
        "bright", "bring", "brisk", "broccoli", "broken", "bronze", "broom", "brother",
        "brown", "brush", "bubble", "buddy", "budget", "buffalo", "build", "bulb",
        "bulk", "bullet", "bundle", "bunker", "burden", "burger", "burst", "bus",
        "business", "busy", "butter", "buyer", "buzz", "cabbage", "cabin", "cable",
        "cactus", "cage", "cake", "call", "calm", "camera", "camp", "can",
        "canal", "cancel", "candy", "cannon", "canoe", "canvas", "canyon", "capable",
        "capital", "captain", "car", "carbon", "card", "cargo", "carpet", "carry",
        "cart", "case", "cash", "casino", "castle", "casual", "cat", "catalog",
        "catch", "category", "cattle", "caught", "cause", "caution", "cave", "ceiling",
        "celery", "cement", "census", "century", "cereal", "certain", "chair", "chalk",
        "champion", "change", "chaos", "chapter", "charge", "chase", "chat", "cheap",
        "check", "cheese", "chef", "cherry", "chest", "chicken", "chief", "child",
        "chimney", "choice", "choose", "chronic", "chuckle", "chunk", "churn", "cigar",
        "cinnamon", "circle", "citizen", "city", "civil", "claim", "clap", "clarify",
        "claw", "clay", "clean", "clerk", "clever", "click", "client", "cliff",
        "climb", "clinic", "clip", "clock", "clog", "close", "cloth", "cloud",
        "clown", "club", "clump", "cluster", "clutch", "coach", "coast", "coconut",
        "code", "coffee", "coil", "coin", "collect", "color", "column", "combine",
        "come", "comfort", "comic", "common", "company", "concert", "conduct", "confirm",
        "congress", "connect", "consider", "control", "convince", "cook", "cool", "copper",
        "copy", "coral", "core", "corn", "correct", "cost", "cotton", "couch",
        "country", "couple", "course", "cousin", "cover", "coyote", "crack", "cradle",
        "craft", "cram", "crane", "crash", "crater", "crawl", "crazy", "cream",
        "credit", "creek", "crew", "cricket", "crime", "crisp", "critic", "crop",
        "cross", "crouch", "crowd", "crucial", "cruel", "cruise", "crumble", "crunch",
        "crush", "cry", "crystal", "cube", "culture", "cup", "cupboard", "curious",
        "current", "curtain", "curve", "cushion", "custom", "cute", "cycle", "dad",
        "damage", "damp", "dance", "danger", "daring", "dash", "daughter", "dawn",
        "day", "deal", "debate", "debris", "decade", "december", "decide", "decline",
        "decorate", "decrease", "deer", "defense", "define", "defy", "degree", "delay",
        "deliver", "demand", "demise", "denial", "dentist", "deny", "depart", "depend",
        "deposit", "depth", "deputy", "derive", "describe", "desert", "design", "desk",
        "despair", "destroy", "detail", "detect", "develop", "device", "devote", "diagram",
        "dial", "diamond", "diary", "dice", "diesel", "diet", "differ", "digital",
        "dignity", "dilemma", "dinner", "dinosaur", "direct", "dirt", "disagree", "discover",
        "disease", "dish", "dismiss", "disorder", "display", "distance", "divert", "divide",
        "divorce", "dizzy", "doctor", "document", "dog", "doll", "dolphin", "domain",
        "donate", "donkey", "donor", "door", "dose", "double", "dove", "draft",
        "dragon", "drama", "drastic", "draw", "dream", "dress", "drift", "drill",
        "drink", "drip", "drive", "drop", "drum", "dry", "duck", "dumb",
        "dune", "during", "dust", "dutch", "duty", "dwarf", "dynamic", "eager",
        "eagle", "early", "earn", "earth", "easily", "east", "easy", "echo",
        "ecology", "economy", "edge", "edit", "educate", "effort", "egg", "eight",
        "either", "elbow", "elder", "electric", "elegant", "element", "elephant", "elevator",
        "elite", "else", "embark", "embody", "embrace", "emerge", "emotion", "employ",
        "empower", "empty", "enable", "enact", "end", "endless", "endorse", "enemy",
        "energy", "enforce", "engage", "engine", "enhance", "enjoy", "enlist", "enough",
        "enrich", "enroll", "ensure", "enter", "entire", "entry", "envelope", "episode",
        "equal", "equip", "era", "erase", "erode", "erosion", "error", "erupt",
        "escape", "essay", "essence", "estate", "eternal", "ethics", "evidence", "evil",
        "evoke", "evolve", "exact", "example", "excess", "exchange", "excite", "exclude",
        "excuse", "execute", "exercise", "exhaust", "exhibit", "exile", "exist", "exit",
        "exotic", "expand", "expect", "expire", "explain", "expose", "express", "extend",
        "extra", "eye", "eyebrow", "fabric", "face", "faculty", "fade", "faint",
        "faith", "fall", "false", "fame", "family", "famous", "fan", "fancy",
        "fantasy", "farm", "fashion", "fat", "fatal", "father", "fatigue", "fault",
        "favorite", "feature", "february", "federal", "fee", "feed", "feel", "female",
        "fence", "festival", "fetch", "fever", "few", "fiber", "fiction", "field",
        "figure", "file", "film", "filter", "final", "find", "fine", "finger",
        "finish", "fire", "firm", "first", "fiscal", "fish", "fit", "fitness",
        "fix", "flag", "flame", "flash", "flat", "flavor", "flee", "flight",
        "flip", "float", "flock", "floor", "flower", "fluid", "flush", "fly",
        "foam", "focus", "fog", "foil", "fold", "follow", "food", "foot",
        "force", "forest", "forget", "fork", "fortune", "forum", "forward", "fossil",
        "foster", "found", "fox", "fragile", "frame", "frequent", "fresh", "friend",
        "fringe", "frog", "front", "frost", "frown", "frozen", "fruit", "fuel",
        "fun", "funny", "furnace", "fury", "future", "gadget", "gain", "galaxy",
        "gallery", "game", "gap", "garage", "garbage", "garden", "garlic", "garment",
        "gas", "gasp", "gate", "gather", "gauge", "gaze", "general", "genius",
        "genre", "gentle", "genuine", "gesture", "ghost", "giant", "gift", "giggle",
        "ginger", "giraffe", "girl", "give", "glad", "glance", "glare", "glass",
        "glide", "glimpse", "globe", "gloom", "glory", "glove", "glow", "glue",
        "goat", "goddess", "gold", "good", "goose", "gorilla", "gospel", "gossip",
        "govern", "gown", "grab", "grace", "grain", "grant", "grape", "grass",
        "gravity", "great", "green", "grid", "grief", "grit", "grocery", "group",
        "grow", "grunt", "guard", "guess", "guide", "guilt", "guitar", "gun",
        "gym", "habit", "hair", "half", "hammer", "hamster", "hand", "happy",
        "harbor", "hard", "harsh", "harvest", "hat", "have", "hawk", "hazard",
        "head", "health", "heart", "heavy", "hedgehog", "height", "hello", "helmet",
        "help", "hen", "hero", "hidden", "high", "hill", "hint", "hip",
        "hire", "history", "hobby", "hockey", "hold", "hole", "holiday", "hollow",
        "home", "honey", "hood", "hope", "horn", "horror", "horse", "hospital",
        "host", "hotel", "hour", "hover", "hub", "huge", "human", "humble",
        "humor", "hundred", "hungry", "hunt", "hurdle", "hurry", "hurt", "husband",
        "hybrid", "ice", "icon", "idea", "identify", "idle", "ignore", "ill",
        "illegal", "illness", "image", "imitate", "immense", "immune", "impact", "impose",
        "improve", "impulse", "inch", "include", "income", "increase", "index", "indicate",
        "indoor", "industry", "infant", "inflict", "inform", "inhale", "inherit", "initial",
        "inject", "injury", "inmate", "inner", "innocent", "input", "inquiry", "insane",
        "insect", "inside", "inspire", "install", "intact", "interest", "into", "invest",
        "invite", "involve", "iron", "island", "isolate", "issue", "item", "ivory",
        "jacket", "jaguar", "jar", "jazz", "jealous", "jeans", "jelly", "jewel",
        "job", "join", "joke", "journey", "joy", "judge", "juice", "jump",
        "jungle", "junior", "junk", "just", "kangaroo", "keen", "keep", "ketchup",
        "key", "kick", "kid", "kidney", "kind", "kingdom", "kiss", "kit",
        "kitchen", "kite", "kitten", "kiwi", "knee", "knife", "knock", "know",
        "lab", "label", "labor", "ladder", "lady", "lake", "lamp", "language",
        "laptop", "large", "later", "latin", "laugh", "laundry", "lava", "law",
        "lawn", "lawsuit", "layer", "lazy", "leader", "leaf", "learn", "leave",
        "lecture", "left", "leg", "legal", "legend", "leisure", "lemon", "lend",
        "length", "lens", "leopard", "lesson", "letter", "level", "liar", "liberty",
        "library", "license", "life", "lift", "light", "like", "limb", "limit",
        "link", "lion", "liquid", "list", "little", "live", "lizard", "load",
        "loan", "lobster", "local", "lock", "logic", "lonely", "long", "loop",
        "lottery", "loud", "lounge", "love", "loyal", "lucky", "luggage", "lumber",
        "lunar", "lunch", "luxury", "lyrics", "machine", "mad", "magic", "magnet",
        "maid", "mail", "main", "major", "make", "mammal", "man", "manage",
        "mandate", "mango", "mansion", "manual", "maple", "marble", "march", "margin",
        "marine", "market", "marriage", "mask", "mass", "master", "match", "material",
        "math", "matrix", "matter", "maximum", "maze", "meadow", "mean", "measure",
        "meat", "mechanic", "medal", "media", "melody", "melt", "member", "memory",
        "mention", "menu", "mercy", "merge", "merit", "merry", "mesh", "message",
        "metal", "method", "middle", "midnight", "milk", "million", "mimic", "mind",
        "minimum", "minor", "minute", "miracle", "mirror", "misery", "miss", "mistake",
        "mix", "mixed", "mixture", "mobile", "model", "modify", "mom", "moment",
        "monitor", "monkey", "monster", "month", "moon", "moral", "more", "morning",
        "mosquito", "mother", "motion", "motor", "mountain", "mouse", "move", "movie",
        "much", "muffin", "mule", "multiply", "muscle", "museum", "mushroom", "music",
        "must", "mutual", "myself", "mystery", "myth", "naive", "name", "napkin",
        "narrow", "nasty", "nation", "nature", "near", "neck", "need", "negative",
        "neglect", "neither", "nephew", "nerve", "nest", "net", "network", "neutral",
        "never", "news", "next", "nice", "night", "noble", "noise", "nominee",
        "noodle", "normal", "north", "nose", "notable", "note", "nothing", "notice",
        "novel", "now", "nuclear", "number", "nurse", "nut", "oak", "obey",
        "object", "oblige", "obscure", "observe", "obtain", "obvious", "occur", "ocean",
        "october", "odor", "off", "offer", "office", "often", "oil", "okay",
        "old", "olive", "olympic", "omit", "once", "one", "onion", "online",
        "only", "open", "opera", "opinion", "oppose", "option", "orange", "orbit",
        "orchard", "order", "ordinary", "organ", "orient", "original", "orphan", "ostrich",
        "other", "outdoor", "outer", "output", "outside", "oval", "oven", "over",
        "own", "owner", "oxygen", "oyster", "ozone", "pact", "paddle", "page",
        "pair", "palace", "palm", "panda", "panel", "panic", "panther", "paper",
        "parade", "parent", "park", "parrot", "party", "pass", "patch", "path",
        "patient", "patrol", "pattern", "pause", "pave", "payment", "peace", "peanut",
        "pear", "peasant", "pelican", "pen", "penalty", "pencil", "people", "pepper",
        "perfect", "permit", "person", "pet", "phone", "photo", "phrase", "physical",
        "piano", "picnic", "picture", "piece", "pig", "pigeon", "pill", "pilot",
        "pink", "pioneer", "pipe", "pistol", "pitch", "pizza", "place", "planet",
        "plastic", "plate", "play", "please", "pledge", "pluck", "plug", "plunge",
        "poem", "poet", "point", "polar", "pole", "police", "pond", "pony",
        "pool", "popular", "portion", "position", "possible", "post", "potato", "pottery",
        "poverty", "powder", "power", "practice", "praise", "predict", "prefer", "prepare",
        "present", "pretty", "prevent", "price", "pride", "primary", "print", "priority",
        "prison", "private", "prize", "problem", "process", "produce", "profit", "program",
        "project", "promote", "proof", "property", "prosper", "protect", "proud", "provide",
        "public", "pudding", "pull", "pulp", "pulse", "pumpkin", "punch", "pupil",
        "puppy", "purchase", "purity", "purpose", "purse", "push", "put", "puzzle",
        "pyramid", "quality", "quantum", "quarter", "question", "quick", "quit", "quiz",
        "quote", "rabbit", "raccoon", "race", "rack", "radar", "radio", "rail",
        "rain", "raise", "rally", "ramp", "ranch", "random", "range", "rapid",
        "rare", "rate", "rather", "raven", "raw", "razor", "ready", "real",
        "reason", "rebel", "rebuild", "recall", "receive", "recipe", "record", "recycle",
        "reduce", "reflect", "reform", "refuse", "region", "regret", "regular", "reject",
        "relax", "release", "relief", "rely", "remain", "remember", "remind", "remove",
        "render", "renew", "rent", "reopen", "repair", "repeat", "replace", "report",
        "require", "rescue", "resemble", "resist", "resource", "response", "result", "retire",
        "retreat", "return", "reunion", "reveal", "review", "reward", "rhythm", "rib",
        "ribbon", "rice", "rich", "ride", "ridge", "rifle", "right", "rigid",
        "ring", "riot", "ripple", "risk", "ritual", "rival", "river", "road",
        "roast", "robot", "robust", "rocket", "romance", "roof", "rookie", "room",
        "rose", "rotate", "rough", "round", "route", "royal", "rubber", "rude",
        "rug", "rule", "run", "runway", "rural", "sad", "saddle", "sadness",
        "safe", "sail", "salad", "salmon", "salon", "salt", "salute", "same",
        "sample", "sand", "satisfy", "satoshi", "sauce", "sausage", "save", "say",
        "scale", "scan", "scare", "scatter", "scene", "scheme", "school", "science",
        "scissors", "scorpion", "scout", "scrap", "screen", "script", "scrub", "sea",
        "search", "season", "seat", "second", "secret", "section", "security", "seed",
        "seek", "segment", "select", "sell", "seminar", "senior", "sense", "sentence",
        "series", "service", "session", "settle", "setup", "seven", "shadow", "shaft",
        "shallow", "share", "shed", "shell", "sheriff", "shield", "shift", "shine",
        "ship", "shiver", "shock", "shoe", "shoot", "shop", "short", "shoulder",
        "shove", "shrimp", "shrug", "shuffle", "shy", "sibling", "sick", "side",
        "siege", "sight", "sign", "silent", "silk", "silly", "silver", "similar",
        "simple", "since", "sing", "siren", "sister", "situate", "six", "size",
        "skate", "sketch", "ski", "skill", "skin", "skirt", "skull", "slab",
        "slam", "sleep", "slender", "slice", "slide", "slight", "slim", "slogan",
        "slot", "slow", "slush", "small", "smart", "smile", "smoke", "smooth",
        "snack", "snake", "snap", "sniff", "snow", "soap", "soccer", "social",
        "sock", "soda", "soft", "solar", "soldier", "solid", "solution", "solve",
        "someone", "song", "soon", "sorry", "sort", "soul", "sound", "soup",
        "source", "south", "space", "spare", "spatial", "spawn", "speak", "special",
        "speed", "spell", "spend", "sphere", "spice", "spider", "spike", "spin",
        "spirit", "split", "spoil", "sponsor", "spoon", "sport", "spot", "spray",
        "spread", "spring", "spy", "square", "squeeze", "squirrel", "stable", "stadium",
        "staff", "stage", "stairs", "stamp", "stand", "start", "state", "stay",
        "steak", "steel", "stem", "step", "stereo", "stick", "still", "sting",
        "stock", "stomach", "stone", "stool", "story", "stove", "strategy", "street",
        "strike", "strong", "struggle", "student", "stuff", "stumble", "style", "subject",
        "submit", "subway", "success", "such", "sudden", "suffer", "sugar", "suggest",
        "suit", "summer", "sun", "sunny", "sunset", "super", "supply", "supreme",
        "sure", "surface", "surge", "surprise", "surround", "survey", "suspect", "sustain",
        "swallow", "swamp", "swap", "swarm", "swear", "sweet", "swift", "swim",
        "swing", "switch", "sword", "symbol", "symptom", "syrup", "system", "table",
        "tackle", "tag", "tail", "talent", "talk", "tank", "tape", "target",
        "task", "taste", "tattoo", "taxi", "teach", "team", "tell", "ten",
        "tenant", "tennis", "tent", "term", "test", "text", "thank", "that",
        "theme", "then", "theory", "there", "they", "thing", "this", "thought",
        "three", "thrive", "throw", "thumb", "thunder", "ticket", "tide", "tiger",
        "tilt", "timber", "time", "tiny", "tip", "tired", "tissue", "title",
        "toast", "tobacco", "today", "toddler", "toe", "together", "toilet", "token",
        "tomato", "tomorrow", "tone", "tongue", "tonight", "tool", "tooth", "top",
        "topic", "topple", "torch", "tornado", "tortoise", "toss", "total", "tourist",
        "toward", "tower", "town", "toy", "track", "trade", "traffic", "tragic",
        "train", "transfer", "trap", "trash", "travel", "tray", "treat", "tree",
        "trend", "trial", "tribe", "trick", "trigger", "trim", "trip", "trophy",
        "trouble", "truck", "true", "truly", "trumpet", "trust", "truth", "try",
        "tube", "tuition", "tumble", "tuna", "tunnel", "turkey", "turn", "turtle",
        "twelve", "twenty", "twice", "twin", "twist", "two", "type", "typical",
        "ugly", "umbrella", "unable", "unaware", "uncle", "uncover", "under", "undo",
        "unfair", "unfold", "unhappy", "uniform", "unique", "unit", "universe", "unknown",
        "unlock", "until", "unusual", "unveil", "update", "upgrade", "uphold", "upon",
        "upper", "upset", "urban", "urge", "usage", "use", "used", "useful",
        "useless", "usual", "utility", "vacant", "vacuum", "vague", "valid", "valley",
        "valve", "van", "vanish", "vapor", "various", "vast", "vault", "vehicle",
        "velvet", "vendor", "venture", "venue", "verb", "verify", "version", "very",
        "vessel", "veteran", "viable", "vibrant", "vicious", "victory", "video", "view",
        "village", "vintage", "violin", "virtual", "virus", "visa", "visit", "visual",
        "vital", "vivid", "vocal", "voice", "void", "volcano", "volume", "vote",
        "voyage", "wage", "wagon", "wait", "walk", "wall", "walnut", "want",
        "warfare", "warm", "warrior", "wash", "wasp", "waste", "water", "wave",
        "way", "wealth", "weapon", "wear", "weasel", "weather", "web", "wedding",
        "weekend", "weird", "welcome", "west", "wet", "whale", "what", "wheat",
        "wheel", "when", "where", "whip", "whisper", "wide", "width", "wife",
        "wild", "will", "win", "window", "wine", "wing", "wink", "winner",
        "winter", "wire", "wisdom", "wise", "wish", "witness", "wolf", "woman",
        "wonder", "wood", "wool", "word", "work", "world", "worry", "worth",
        "wrap", "wreck", "wrestle", "wrist", "write", "wrong", "yard", "year",
        "yellow", "you", "young", "youth", "zebra", "zero", "zone", "zoo"
    ]
    
    def __init__(self, history_file: str = "seed_history.json"):
        """
        Инициализация генератора
        
        Args:
            history_file: Файл для хранения истории сгенерированных фраз
        """
        self.history_file = history_file
        self.generated_phrases = self._load_history()
        
    def _load_history(self) -> set:
        """Загрузить историю сгенерированных фраз"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return set(data.get('phrases', []))
            except (json.JSONDecodeError, IOError):
                return set()
        return set()
    
    def _save_history(self):
        """Сохранить историю в файл"""
        try:
            data = {
                'phrases': list(self.generated_phrases),
                'total_generated': len(self.generated_phrases),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Ошибка сохранения истории: {e}")
    
    def _calculate_checksum(self, entropy: bytes) -> int:
        """Вычислить контрольную сумму для энтропии"""
        hash_bytes = hashlib.sha256(entropy).digest()
        checksum_bits = len(entropy) * 8 // 32
        return hash_bytes[0] >> (8 - checksum_bits)
    
    def _entropy_to_mnemonic(self, entropy: bytes, word_count: int) -> List[str]:
        """Преобразовать энтропию в мнемоническую фразу"""
        checksum = self._calculate_checksum(entropy)
        checksum_bits = len(entropy) * 8 // 32
        
        # Объединяем энтропию и контрольную сумму в битовую строку
        binary_entropy = ''.join(format(byte, '08b') for byte in entropy)
        binary_checksum = format(checksum, f'0{checksum_bits}b')
        binary_combined = binary_entropy + binary_checksum
        
        # Разбиваем на группы по 11 бит и конвертируем в индексы слов
        words = []
        for i in range(0, len(binary_combined), 11):
            if i + 11 <= len(binary_combined):
                word_bits = binary_combined[i:i+11]
                word_index = int(word_bits, 2)
                words.append(self.WORDLIST[word_index])
        
        return words[:word_count]
    
    def generate_seed_phrase(self, word_count: int = 12) -> Optional[List[str]]:
        """
        Генерация уникальной seed фразы
        
        Args:
            word_count: Количество слов (12, 15, 18, 21, 24)
            
        Returns:
            Список слов seed фразы или None если не найдена уникальная
        """
        if word_count not in [12, 15, 18, 21, 24]:
            raise ValueError("Количество слов должно быть 12, 15, 18, 21 или 24")
        
        # Определяем размер энтропии
        entropy_bits = {12: 128, 15: 160, 18: 192, 21: 224, 24: 256}
        entropy_size = entropy_bits[word_count] // 8
        
        # Попытки генерации уникальной фразы
        max_attempts = 1000
        for attempt in range(max_attempts):
            # Генерируем криптографически стойкую энтропию
            entropy = secrets.token_bytes(entropy_size)
            
            # Преобразуем в мнемоническую фразу
            words = self._entropy_to_mnemonic(entropy, word_count)
            phrase_str = ' '.join(words)
            
            # Проверяем уникальность
            if phrase_str not in self.generated_phrases:
                self.generated_phrases.add(phrase_str)
                self._save_history()
                return words
        
        # Если не удалось найти уникальную фразу
        return None
    
    def is_valid_bip39(self, words: List[str]) -> bool:
        """Проверка валидности BIP39 фразы"""
        if len(words) not in [12, 15, 18, 21, 24]:
            return False
        
        # Проверяем, что все слова есть в словаре
        for word in words:
            if word not in self.WORDLIST:
                return False
        
        # Преобразуем слова в битовую строку
        binary_combined = ''
        for word in words:
            word_index = self.WORDLIST.index(word)
            binary_combined += format(word_index, '011b')
        
        # Разделяем энтропию и контрольную сумму
        entropy_bits = len(words) * 11 - len(words) * 11 // 33
        entropy_binary = binary_combined[:entropy_bits]
        checksum_binary = binary_combined[entropy_bits:]
        
        # Преобразуем энтропию в байты
        entropy_bytes = []
        for i in range(0, len(entropy_binary), 8):
            byte_bits = entropy_binary[i:i+8]
            if len(byte_bits) == 8:
                entropy_bytes.append(int(byte_bits, 2))
        
        entropy = bytes(entropy_bytes)
        
        # Вычисляем ожидаемую контрольную сумму
        expected_checksum = self._calculate_checksum(entropy)
        checksum_bits = len(checksum_binary)
        expected_checksum_binary = format(expected_checksum, f'0{checksum_bits}b')
        
        return checksum_binary == expected_checksum_binary
    
    def generate_qr_code(self, phrase: List[str]) -> bytes:
        """Генерация QR кода для seed фразы"""
        phrase_str = ' '.join(phrase)
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(phrase_str)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику генерации"""
        return {
            'total_generated': len(self.generated_phrases),
            'history_file': self.history_file,
            'unique_phrases': len(self.generated_phrases)
        }


class BIP39GUI:
    """GUI интерфейс для BIP39 генератора"""
    
    def __init__(self):
        self.generator = BIP39Generator()
        self.root = tk.Tk()
        self.root.title("BIP39 Seed Phrase Generator v2.0")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Переменные
        self.word_count_var = tk.StringVar(value="12")
        self.current_phrase = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Конфигурация сетки
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="BIP39 Seed Phrase Generator", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Выбор количества слов
        ttk.Label(main_frame, text="Количество слов:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        word_frame = ttk.Frame(main_frame)
        word_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        for i, count in enumerate([12, 15, 18, 21, 24]):
            ttk.Radiobutton(word_frame, text=str(count), variable=self.word_count_var, 
                           value=str(count)).grid(row=0, column=i, padx=5)
        
        # Кнопка генерации
        generate_btn = ttk.Button(main_frame, text="Генерировать новую фразу", 
                                 command=self.generate_phrase, style='Accent.TButton')
        generate_btn.grid(row=2, column=0, columnspan=3, pady=20)
        
        # Текстовое поле для отображения фразы
        ttk.Label(main_frame, text="Сгенерированная фраза:").grid(row=3, column=0, sticky=tk.W)
        
        self.phrase_text = scrolledtext.ScrolledText(main_frame, height=4, width=80, 
                                                    font=('Courier', 12), wrap=tk.WORD)
        self.phrase_text.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Кнопки действий
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        ttk.Button(btn_frame, text="Копировать", 
                  command=self.copy_phrase).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Показать QR код", 
                  command=self.show_qr_code).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Проверить фразу", 
                  command=self.validate_phrase).grid(row=0, column=2, padx=5)
        ttk.Button(btn_frame, text="Очистить", 
                  command=self.clear_phrase).grid(row=0, column=3, padx=5)
        
        # Поле для проверки фразы
        ttk.Label(main_frame, text="Проверить существующую фразу:").grid(row=6, column=0, sticky=tk.W, pady=(20, 5))
        
        self.validate_text = scrolledtext.ScrolledText(main_frame, height=3, width=80, 
                                                      font=('Courier', 11))
        self.validate_text.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Результат валидации
        self.validation_result = ttk.Label(main_frame, text="", font=('Arial', 10))
        self.validation_result.grid(row=8, column=0, columnspan=3, pady=5)
        
        # Статистика
        stats_frame = ttk.LabelFrame(main_frame, text="Статистика", padding="10")
        stats_frame.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        
        self.stats_label = ttk.Label(stats_frame, text="")
        self.stats_label.grid(row=0, column=0, sticky=tk.W)
        
        self.update_stats()
        
        # Информация
        info_frame = ttk.LabelFrame(main_frame, text="Информация", padding="10")
        info_frame.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        info_text = ("BIP39 - стандарт для генерации мнемонических фраз.\n"
                    "Каждая фраза проверяется на уникальность.\n"
                    "Сохраняйте фразы в безопасном месте!")
        
        ttk.Label(info_frame, text=info_text, font=('Arial', 9), 
                 foreground='gray').grid(row=0, column=0, sticky=tk.W)
    
    def generate_phrase(self):
        """Генерация новой фразы"""
        try:
            word_count = int(self.word_count_var.get())
            phrase = self.generator.generate_seed_phrase(word_count)
            
            if phrase:
                self.current_phrase = phrase
                phrase_str = ' '.join(phrase)
                
                self.phrase_text.delete(1.0, tk.END)
                self.phrase_text.insert(1.0, phrase_str)
                
                self.update_stats()
                messagebox.showinfo("Успех", f"Сгенерирована новая уникальная фраза из {word_count} слов!")
            else:
                messagebox.showerror("Ошибка", "Не удалось сгенерировать уникальную фразу после 1000 попыток")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка генерации: {str(e)}")
    
    def copy_phrase(self):
        """Копирование фразы в буфер обмена"""
        if self.current_phrase:
            phrase_str = ' '.join(self.current_phrase)
            self.root.clipboard_clear()
            self.root.clipboard_append(phrase_str)
            messagebox.showinfo("Скопировано", "Фраза скопирована в буфер обмена")
        else:
            messagebox.showwarning("Предупреждение", "Нет фразы для копирования")
    
    def show_qr_code(self):
        """Показать QR код фразы"""
        if not self.current_phrase:
            messagebox.showwarning("Предупреждение", "Нет фразы для создания QR кода")
            return
        
        try:
            qr_bytes = self.generator.generate_qr_code(self.current_phrase)
            
            # Создаем новое окно для QR кода
            qr_window = tk.Toplevel(self.root)
            qr_window.title("QR код seed фразы")
            qr_window.geometry("400x450")
            
            # Загружаем и отображаем изображение
            qr_image = Image.open(io.BytesIO(qr_bytes))
            qr_photo = ImageTk.PhotoImage(qr_image)
            
            qr_label = ttk.Label(qr_window, image=qr_photo)
            qr_label.image = qr_photo  # Сохраняем ссылку
            qr_label.pack(pady=20)
            
            warning_text = ("⚠️ ВНИМАНИЕ: Не делитесь этим QR кодом!\n"
                          "Он содержит вашу приватную seed фразу.")
            ttk.Label(qr_window, text=warning_text, foreground='red', 
                     font=('Arial', 10, 'bold')).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка создания QR кода: {str(e)}")
    
    def validate_phrase(self):
        """Проверка введенной фразы"""
        phrase_text = self.validate_text.get(1.0, tk.END).strip()
        
        if not phrase_text:
            self.validation_result.config(text="Введите фразу для проверки", foreground='orange')
            return
        
        try:
            words = phrase_text.split()
            is_valid = self.generator.is_valid_bip39(words)
            
            if is_valid:
                self.validation_result.config(text="✓ Фраза валидна", foreground='green')
            else:
                self.validation_result.config(text="✗ Фраза невалидна", foreground='red')
                
        except Exception as e:
            self.validation_result.config(text=f"Ошибка проверки: {str(e)}", foreground='red')
    
    def clear_phrase(self):
        """Очистка всех полей"""
        self.phrase_text.delete(1.0, tk.END)
        self.validate_text.delete(1.0, tk.END)
        self.validation_result.config(text="")
        self.current_phrase = []
    
    def update_stats(self):
        """Обновление статистики"""
        stats = self.generator.get_stats()
        stats_text = f"Всего сгенерировано уникальных фраз: {stats['total_generated']}"
        self.stats_label.config(text=stats_text)
    
    def run(self):
        """Запуск GUI"""
        self.root.mainloop()


def main():
    """Основная функция"""
    try:
        # Проверяем наличие необходимых библиотек
        import PIL
        print("BIP39 Seed Phrase Generator v2.0")
        print("Загрузка...")
        
        # Создаем и запускаем GUI
        app = BIP39GUI()
        app.run()
        
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Установите необходимые библиотеки:")
        print("pip install pillow qrcode[pil]")
    except Exception as e:
        print(f"Критическая ошибка: {e}")


if __name__ == "__main__":
    main()
