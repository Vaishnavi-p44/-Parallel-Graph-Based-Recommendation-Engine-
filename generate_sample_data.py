"""
Data generator for the Parallel Graph-Based Recommendation Engine.
Generates realistic, rich synthetic datasets for testing and demonstration:
1. MovieLens-style Movies Dataset (users, movies, ratings, genres)
2. E-Commerce Tech Gadgets Dataset (users, products, ratings, categories)
"""

import os
import random
import pandas as pd
import numpy as np

def create_movie_dataset(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    # 120 Curated realistic movie titles across distinct genres
    movies_data = [
        # Sci-Fi / Space
        ("M101", "Interstellar Odyssey", "Sci-Fi, Adventure", 2014, "Explorers travel through a wormhole in search of a new home.", 4.8),
        ("M102", "Cyber Matrix 2099", "Sci-Fi, Action", 2019, "A hacker discovers the virtual nature of human reality.", 4.6),
        ("M103", "Nebula Rising", "Sci-Fi, Drama", 2021, "First contact with an enigmatic deep space anomaly.", 4.4),
        ("M104", "Quantum Paradox", "Sci-Fi, Thriller", 2018, "Physicists accidentally unravel the fabric of parallel timelines.", 4.7),
        ("M105", "Solaris Station", "Sci-Fi, Mystery", 2015, "Cosmonauts investigate bizarre psychological phenomena on an orbital station.", 4.3),
        ("M106", "Starlight Vanguard", "Sci-Fi, Adventure", 2022, "Intergalactic bounty hunters protect an ancient star map.", 4.5),
        ("M107", "The Martian Horizon", "Sci-Fi, Drama", 2017, "An astronaut strives to survive isolation on the red planet.", 4.8),
        ("M108", "Android Genesis", "Sci-Fi, Thriller", 2020, "Synthetic humans demand autonomy in a futuristic metropolis.", 4.2),
        ("M109", "Hyperdrive Chronicles", "Sci-Fi, Action", 2023, "Rogue pilots race against time to intercept a planetary defense weapon.", 4.4),
        ("M110", "Cosmic Echoes", "Sci-Fi, Mystery", 2016, "Radio astronomers decode a repeating signal from the Andromeda galaxy.", 4.5),

        # Action / Thriller
        ("M111", "Apex Predator", "Action, Thriller", 2018, "An elite operative navigates a deadly conspiracy in Berlin.", 4.6),
        ("M112", "Velocity Surge", "Action, Crime", 2020, "Undercover drivers infiltrate a high-stakes heist syndicate.", 4.3),
        ("M113", "Midnight Protocol", "Action, Thriller", 2019, "A black-ops team must recover stolen biometric encryption keys.", 4.5),
        ("M114", "Shadow Citadel", "Action, Mystery", 2021, "Mercenaries storm a fortified bunker hidden in the Alps.", 4.2),
        ("M115", "The Iron Courier", "Action, Adventure", 2016, "A hardened transport specialist delivers a dangerous package across war zones.", 4.4),
        ("M116", "Subzero Extraction", "Action, Thriller", 2022, "Rescue mission in the unforgiving Arctic permafrost.", 4.7),
        ("M117", "Rogue Syndicate", "Action, Crime", 2017, "Detectives wage an unsanctioned urban war on a ruthless cartel.", 4.5),
        ("M118", "Thunderbolt Strike", "Action, War", 2015, "Fighter jet squadrons defend an island sanctuary.", 4.1),
        ("M119", "Vengeance Creed", "Action, Drama", 2023, "A martial artist seeks justice for a betrayed brotherhood.", 4.6),
        ("M120", "Steel Horizon", "Action, Thriller", 2018, "Naval commandos board a hijacked supersonic destroyer.", 4.3),

        # Drama / Biography / History
        ("M121", "The Pianist's Echo", "Drama, Music", 2016, "A classical virtuoso confronts personal tragedy in 1950s Vienna.", 4.9),
        ("M122", "Crown of Thorns", "Drama, History", 2019, "The untold political intrigues of a Renaissance monarchy.", 4.7),
        ("M123", "Autumn in Kyoto", "Drama, Romance", 2021, "Two travelers rediscover meaning amid traditional tea houses.", 4.6),
        ("M124", "Silicon Titans", "Drama, Biography", 2018, "The competitive founding of modern personal computing.", 4.8),
        ("M125", "The Silent Witness", "Drama, Crime", 2017, "A rural prosecutor battles systemic corruption in a small mining town.", 4.5),
        ("M126", "Letters from Normandy", "Drama, War", 2014, "Family correspondence spanning the turbulent battlefields of WWII.", 4.8),
        ("M127", "The Golden Canvas", "Drama, Biography", 2020, "An Impressionist painter defies societal norms to create masterworks.", 4.4),
        ("M128", "Winter Harbor", "Drama, Mystery", 2022, "A seaside community reckons with secrets uncovered after a severe storm.", 4.3),
        ("M129", "Symphony of Hope", "Drama, Music", 2015, "A youth orchestra overcomes poverty to compete on the world stage.", 4.7),
        ("M130", "The Last Architect", "Drama, History", 2023, "Designing the world's most daring bridge in the 1920s.", 4.5),

        # Comedy / Romance
        ("M131", "Cupid's Algorithm", "Comedy, Romance", 2021, "Dating app engineers fall in love while testing compatibility bugs.", 4.4),
        ("M132", "Meet the In-Laws 2.0", "Comedy, Family", 2019, "A chaotic weekend wedding in Tuscany filled with culinary mishaps.", 4.2),
        ("M133", "Lost in Barcelona", "Romance, Adventure", 2018, "A backpacker and an architect form an unexpected connection.", 4.5),
        ("M134", "The Startup Fiasco", "Comedy", 2020, "Four eccentric roommates pitch absurd ideas to venture capitalists.", 4.6),
        ("M135", "Coffee & Croissants", "Romance, Comedy", 2017, "Daily morning conversations spark a sweet romance in Paris.", 4.3),
        ("M136", "Almost Royal", "Comedy, Romance", 2022, "A modest teacher discovers an eccentric claim to a tiny European principality.", 4.1),
        ("M137", "Game Night Chaos", "Comedy, Mystery", 2018, "A friendly board game evening turns into an outrageous murder mystery.", 4.7),
        ("M138", "The Accidental Guru", "Comedy", 2023, "A frustrated office clerk becomes a viral self-help sensation overnight.", 4.3),
        ("M139", "Sunset in Santorini", "Romance, Drama", 2016, "Reuniting with a childhood friend during a Mediterranean summer.", 4.5),
        ("M140", "Wedding Crashers Club", "Comedy, Romance", 2021, "Best friends make a habit of uninvited celebration dining.", 4.2),

        # Animation / Fantasy / Adventure
        ("M141", "Spirit of the Forest", "Animation, Fantasy", 2019, "A young guardian protects sacred woodland spirits from an ancient blight.", 4.9),
        ("M142", "Dragon Whisperer", "Fantasy, Adventure", 2018, "An outcast youth discovers an empathetic bond with mythical beasts.", 4.8),
        ("M143", "Clockwork City", "Animation, Sci-Fi", 2021, "An apprentice clockmaker uncovers gear-driven secrets in a floating city.", 4.7),
        ("M144", "The Enchanted Quill", "Fantasy, Family", 2017, "Whatever a young scribe sketches comes magically to life.", 4.6),
        ("M145", "Whale Song Journey", "Animation, Adventure", 2020, "A marine biologist and a speaking humpback whale navigate oceanic currents.", 4.5),
        ("M146", "Rune Knight", "Fantasy, Action", 2022, "A paladin seeks elemental gems to seal a breach in the celestial wall.", 4.4),
        ("M147", "The Celestial Compass", "Adventure, Fantasy", 2016, "A daring expedition tracks the celestial needle to hidden islands.", 4.7),
        ("M148", "Goblins & Gliders", "Animation, Comedy", 2023, "Playful forest goblins build aeronautical inventions out of twigs.", 4.3),
        ("M149", "The Phoenix Ring", "Fantasy, Adventure", 2015, "An ancient artifact grants the power of rebirth at great cost.", 4.6),
        ("M150", "Isle of Whispers", "Fantasy, Mystery", 2020, "Navigating a foggy archipelago where memories echo in the wind.", 4.5),

        # Horror / Mystery / Psychological
        ("M151", "The Whispering Walls", "Horror, Mystery", 2017, "A family moves into a heritage estate with a sinister acoustic past.", 4.3),
        ("M152", "Shadow over Innsbrook", "Horror, Thriller", 2020, "An investigative journalist discovers occult rites in a secluded harbor.", 4.4),
        ("M153", "Fractured Mind", "Thriller, Mystery", 2019, "A detective suffering from memory lapses must solve a locked-room puzzle.", 4.8),
        ("M154", "Labyrinth of Silence", "Mystery, Drama", 2018, "Unraveling an encrypted diary left behind by a vanished mathematician.", 4.6),
        ("M155", "The Crimson Manor", "Horror, Thriller", 2021, "Guests at an antique mansion disappear one by one during a blackout.", 4.2),
        ("M156", "Echoes from the Abyss", "Horror, Sci-Fi", 2022, "Deep sea drillers awaken a dormant prehistoric consciousness.", 4.5),
        ("M157", "Paranoia Alley", "Thriller, Crime", 2016, "A surveillance technician witnesses an unrecorded event from his monitors.", 4.7),
        ("M158", "The Dollmaker's Curse", "Horror, Mystery", 2023, "Porcelain dolls appear to mirror real-life villagers' fateful decisions.", 4.1),
        ("M159", "Ghost Frequency", "Mystery, Sci-Fi", 2019, "A radio podcast host picks up transmissions from decades in the future.", 4.4),
        ("M160", "Nocturne Phantoms", "Horror, Drama", 2015, "A night-shift curator experiences inexplicable nocturnal visions.", 4.3),
    ]

    # Generate additional items up to 100 items to have rich variety
    genres_pool = [
        ("Sci-Fi", ["Cosmic Gate", "Warp Dimension", "Titan Colony", "Quantum Core", "Singularity Call", "Vortex 9"]),
        ("Action", ["Terminal Breach", "Direct Impact", "Overdrive Patrol", "Iron Battalion", "Rapid Recon", "Urban Siege"]),
        ("Drama", ["The Glass Garden", "Fading Echo", "A Father's Vow", "The River Crossing", "Threads of Fate", "Golden Fields"]),
        ("Comedy", ["Crazy Reunion", "Dog Days of Summer", "Office Pranks", "Bloopers & Bagels", "Weekend Royale", "The Food Truck"]),
        ("Fantasy", ["Moonlight Spire", "Blade of Eldoria", "Crystal Cavern", "The Mage's Oath", "Sunken Citadel", "Forest of Runes"]),
        ("Mystery", ["Blind Spot", "The Cipher Note", "Smoke & Glass", "Midnight Alibi", "Vanishing Act", "Foggy Pier"]),
    ]

    item_id_counter = 161
    for genre, titles in genres_pool:
        for t in titles:
            movies_data.append((
                f"M{item_id_counter}",
                t,
                f"{genre}, Thriller" if genre in ["Sci-Fi", "Action"] else f"{genre}, Drama",
                random.randint(2012, 2024),
                f"A captivating story exploring {genre.lower()} themes and unforgettable characters.",
                round(random.uniform(3.9, 4.9), 1)
            ))
            item_id_counter += 1

    df_meta = pd.DataFrame(movies_data, columns=["item_id", "title", "genres", "release_year", "description", "rating_avg"])
    meta_path = os.path.join(output_dir, "movies_metadata.csv")
    df_meta.to_csv(meta_path, index=False)
    print(f"Generated {len(df_meta)} movies in {meta_path}")

    # Generate interactions with realistic preference clusters
    # 100 Users:
    # Users U1-U25 prefer Sci-Fi & Action
    # Users U26-U50 prefer Drama & Romance
    # Users U51-U75 prefer Animation & Fantasy
    # Users U76-U100 have broad eclectic tastes
    users = [f"U{i:03d}" for i in range(1, 101)]
    interactions = []

    # Map items to primary genre
    scifi_action_items = df_meta[df_meta["genres"].str.contains("Sci-Fi|Action")]["item_id"].tolist()
    drama_romance_items = df_meta[df_meta["genres"].str.contains("Drama|Romance|Comedy")]["item_id"].tolist()
    fantasy_items = df_meta[df_meta["genres"].str.contains("Fantasy|Animation|Adventure")]["item_id"].tolist()
    all_items = df_meta["item_id"].tolist()

    for idx, user in enumerate(users):
        u_num = idx + 1
        # Number of interactions per user: between 15 and 35
        n_interactions = random.randint(18, 36)

        if u_num <= 25:
            # Sci-Fi / Action focused
            pref_pool = scifi_action_items
            other_pool = [it for it in all_items if it not in pref_pool]
            n_pref = int(n_interactions * 0.75)
            n_other = n_interactions - n_pref
            chosen = random.sample(pref_pool, min(n_pref, len(pref_pool))) + random.sample(other_pool, min(n_other, len(other_pool)))
        elif u_num <= 50:
            # Drama / Romance focused
            pref_pool = drama_romance_items
            other_pool = [it for it in all_items if it not in pref_pool]
            n_pref = int(n_interactions * 0.75)
            n_other = n_interactions - n_pref
            chosen = random.sample(pref_pool, min(n_pref, len(pref_pool))) + random.sample(other_pool, min(n_other, len(other_pool)))
        elif u_num <= 75:
            # Fantasy / Adventure focused
            pref_pool = fantasy_items
            other_pool = [it for it in all_items if it not in pref_pool]
            n_pref = int(n_interactions * 0.75)
            n_other = n_interactions - n_pref
            chosen = random.sample(pref_pool, min(n_pref, len(pref_pool))) + random.sample(other_pool, min(n_other, len(other_pool)))
        else:
            # Eclectic
            chosen = random.sample(all_items, n_interactions)

        for item_id in set(chosen):
            rating = round(random.uniform(3.5, 5.0), 1)
            # Timestamp formatted as unix or standard date
            timestamp = 1672531199 + random.randint(1000, 30000000)
            interactions.append((user, item_id, rating, timestamp))

    df_inter = pd.DataFrame(interactions, columns=["user_id", "item_id", "rating", "timestamp"])
    inter_path = os.path.join(output_dir, "movies_interactions.csv")
    df_inter.to_csv(inter_path, index=False)
    print(f"Generated {len(df_inter)} interactions for {len(users)} users in {inter_path}")

def create_ecommerce_dataset(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(99)
    np.random.seed(99)

    products = [
        # Audio
        ("P201", "AcousticPro Wireless Noise-Cancelling Headphones", "Audio", 249.99, "Over-ear active noise cancelling with 40h battery."),
        ("P202", "BassPulse Mini Bluetooth Speaker", "Audio", 59.99, "Waterproof IPX7 outdoor speaker with rich bass."),
        ("P203", "StudioMaster Reference Monitors", "Audio", 399.00, "Near-field studio powered monitors with flat frequency response."),
        ("P204", "TrueSonic ANC Earbuds", "Audio", 129.99, "Compact wireless earbuds with spatial audio and wireless charging."),
        ("P205", "VocalStream USB Condenser Microphone", "Audio", 89.99, "Cardioid studio mic for podcasting and streaming."),

        # Computing / Tech
        ("P206", "UltraSlim 14-inch OLED Laptop", "Computing", 1099.99, "Powered by ultra-fast processor, 32GB RAM, 1TB NVMe."),
        ("P207", "QuantumMechanical Ergonomic Keyboard", "Computing", 149.99, "Hot-swappable tactile switches with RGB per-key backlighting."),
        ("P208", "PrecisionFlow Wireless Mouse", "Computing", 79.99, "High-precision sensor with ergonomic thumb rest and dual Bluetooth."),
        ("P209", "ThunderBolt 4 Multi-Port Docking Station", "Computing", 179.99, "Dual 4K display output, 100W power delivery, 10Gbps USB-C."),
        ("P210", "SwiftDrive 2TB Portable SSD", "Computing", 159.99, "Shock-resistant aluminum enclosure with 2000 MB/s read speed."),

        # Gaming
        ("P211", "TitanForge RTX Gaming Desktop", "Gaming", 1899.00, "Water-cooled gaming rig optimized for 4K ray-traced gaming."),
        ("P212", "HyperView 27-inch 165Hz Curved Monitor", "Gaming", 299.99, "QHD 1ms response time gaming display with FreeSync."),
        ("P213", "TacticalStrike Wireless Gamepad", "Gaming", 69.99, "Hall-effect joysticks, hair triggers, and remappable back paddles."),
        ("P214", "AeroComfort Ergonomic Gaming Chair", "Gaming", 289.99, "Breathable mesh back with lumbar cushion and 4D armrests."),
        ("P215", "RGB Prism Extended Desk Mat", "Gaming", 29.99, "Micro-textured cloth surface with 14 customizable lighting modes."),

        # Smart Home / Wearables
        ("P216", "AuraFit Health & Fitness Smartwatch", "Wearables", 199.99, "Continuous SpO2, heart rate, ECG, and multi-day GPS tracking."),
        ("P217", "GlowSmart Ambience LED Light Strip", "Smart Home", 39.99, "Wi-Fi enabled sync-to-music smart RGBIC lighting strip."),
        ("P218", "PureBreeze HEPA Smart Air Purifier", "Smart Home", 149.99, "Filters 99.97% of airborne allergens with smartphone app control."),
        ("P219", "ThermoSense Wi-Fi Smart Thermostat", "Smart Home", 169.99, "Energy saving adaptive schedule with remote room sensors."),
        ("P220", "VisionSecure 2K Outdoor Security Camera", "Smart Home", 89.99, "Solar-powered weatherproof camera with color night vision."),
    ]

    # Expand to 60 products across categories
    categories = ["Audio", "Computing", "Gaming", "Wearables", "Smart Home"]
    for cat in categories:
        for idx in range(1, 9):
            p_id = f"P{220 + len(products) - 19}"
            title = f"{cat} Pro Series Model {idx}"
            price = round(random.uniform(25.0, 450.0), 2)
            products.append((p_id, title, cat, price, f"High-performance {cat.lower()} accessory designed for enthusiasts."))

    df_meta = pd.DataFrame(products, columns=["item_id", "title", "category", "price", "description"])
    meta_path = os.path.join(output_dir, "ecommerce_metadata.csv")
    df_meta.to_csv(meta_path, index=False)
    print(f"Generated {len(df_meta)} products in {meta_path}")

    # Generate interactions for 80 users
    users = [f"Cust_{i:03d}" for i in range(1, 81)]
    interactions = []
    all_items = df_meta["item_id"].tolist()

    for idx, user in enumerate(users):
        n_items = random.randint(10, 24)
        chosen = random.sample(all_items, n_items)
        for it in chosen:
            rating = round(random.uniform(3.0, 5.0), 1)
            interactions.append((user, it, rating, 1680000000 + random.randint(100, 10000000)))

    df_inter = pd.DataFrame(interactions, columns=["user_id", "item_id", "rating", "timestamp"])
    inter_path = os.path.join(output_dir, "ecommerce_interactions.csv")
    df_inter.to_csv(inter_path, index=False)
    print(f"Generated {len(df_inter)} interactions for {len(users)} customers in {inter_path}")

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    create_movie_dataset(out_dir)
    create_ecommerce_dataset(out_dir)
    print("Sample datasets created successfully!")
