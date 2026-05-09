alter table public.profiles
  add column if not exists appearance_settings jsonb not null default '{}'::jsonb;
