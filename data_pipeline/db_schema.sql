-- AntID metadata schema (PostgreSQL)
--
-- Apply with:  psql "$DATABASE_URL" -f db_schema.sql
--
-- Notes on UNIQUE constraints:
--   * species.slug          — natural key used by upsert_species() ON CONFLICT (slug)
--   * images.storage_path    — natural key used by insert_image()   ON CONFLICT (storage_path)
--   Both are required for the pipeline's idempotent re-runs to work.

CREATE TABLE IF NOT EXISTS species (
    id          SERIAL PRIMARY KEY,
    taxon_id    INTEGER UNIQUE,             -- iNat Formicidae taxon id; resolved at runtime
    slug        TEXT    UNIQUE NOT NULL,    -- url-safe scientific name, canonical join key
    name        TEXT    NOT NULL,           -- scientific name "Genus species"
    common_name TEXT,
    class_idx   INTEGER UNIQUE              -- contiguous 0..N-1 label index, set at train time
);

CREATE TABLE IF NOT EXISTS images (
    id           SERIAL PRIMARY KEY,
    species_id   INTEGER NOT NULL REFERENCES species(id) ON DELETE CASCADE,
    source       TEXT    NOT NULL,          -- 'inat' | 'antweb' | 'gbif'
    storage_path TEXT    UNIQUE NOT NULL,   -- gs:// or s3:// URI of the cleaned image
    split        TEXT    NOT NULL DEFAULT 'train',  -- 'train' | 'val'
    width        INTEGER,
    height       INTEGER,
    lat          DOUBLE PRECISION,          -- observation latitude (nullable)
    lon          DOUBLE PRECISION,          -- observation longitude (nullable)
    created_at   TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT images_source_chk CHECK (source IN ('inat', 'antweb', 'gbif')),
    CONSTRAINT images_split_chk  CHECK (split  IN ('train', 'val'))
);

CREATE INDEX IF NOT EXISTS idx_images_species ON images(species_id);
CREATE INDEX IF NOT EXISTS idx_images_split   ON images(split);

CREATE TABLE IF NOT EXISTS training_runs (
    id            SERIAL PRIMARY KEY,
    started_at    TIMESTAMP,
    finished_at   TIMESTAMP,
    config        JSONB,
    top1_acc      FLOAT,
    top3_acc      FLOAT,
    artifact_path TEXT
);
