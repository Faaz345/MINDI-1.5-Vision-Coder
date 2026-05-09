create extension if not exists pgcrypto;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  first_name text,
  last_name text,
  full_name text,
  phone text,
  avatar_url text,
  plan text not null default 'free',
  appearance_settings jsonb not null default '{}'::jsonb,
  onboarding_completed boolean not null default false,
  last_seen_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

drop policy if exists "Users can read own profile" on public.profiles;
create policy "Users can read own profile"
on public.profiles
for select
to authenticated
using ((select auth.uid()) = id);

drop policy if exists "Users can update own profile" on public.profiles;
create policy "Users can update own profile"
on public.profiles
for update
to authenticated
using ((select auth.uid()) = id)
with check ((select auth.uid()) = id);

drop trigger if exists set_profiles_updated_at on public.profiles;
create trigger set_profiles_updated_at
before update on public.profiles
for each row execute function public.set_updated_at();

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (
    id,
    email,
    first_name,
    last_name,
    full_name,
    phone,
    avatar_url
  )
  values (
    new.id,
    new.email,
    new.raw_user_meta_data ->> 'first_name',
    new.raw_user_meta_data ->> 'last_name',
    coalesce(new.raw_user_meta_data ->> 'full_name', new.raw_user_meta_data ->> 'name'),
    new.raw_user_meta_data ->> 'phone',
    new.raw_user_meta_data ->> 'avatar_url'
  )
  on conflict (id) do update set
    email = excluded.email,
    first_name = excluded.first_name,
    last_name = excluded.last_name,
    full_name = excluded.full_name,
    phone = excluded.phone,
    avatar_url = excluded.avatar_url,
    updated_at = now();

  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
after insert on auth.users
for each row execute function public.handle_new_user();

create table if not exists public.projects (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  name text not null default 'Untitled',
  description text,
  files jsonb not null default '{}'::jsonb,
  design_settings jsonb not null default '{}'::jsonb,
  source_prompt text,
  preview_html text,
  status text not null default 'active',
  metadata jsonb not null default '{}'::jsonb,
  last_opened_at timestamptz not null default now(),
  archived_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists projects_owner_id_updated_at_idx
on public.projects (owner_id, updated_at desc);

create index if not exists projects_owner_active_last_opened_idx
on public.projects (owner_id, last_opened_at desc)
where archived_at is null;

create index if not exists projects_owner_status_idx
on public.projects (owner_id, status);

alter table public.projects enable row level security;

drop policy if exists "Users can read own projects" on public.projects;
create policy "Users can read own projects"
on public.projects
for select
to authenticated
using ((select auth.uid()) = owner_id);

drop policy if exists "Users can create own projects" on public.projects;
create policy "Users can create own projects"
on public.projects
for insert
to authenticated
with check ((select auth.uid()) = owner_id);

drop policy if exists "Users can update own projects" on public.projects;
create policy "Users can update own projects"
on public.projects
for update
to authenticated
using ((select auth.uid()) = owner_id)
with check ((select auth.uid()) = owner_id);

drop policy if exists "Users can delete own projects" on public.projects;
create policy "Users can delete own projects"
on public.projects
for delete
to authenticated
using ((select auth.uid()) = owner_id);

drop trigger if exists set_projects_updated_at on public.projects;
create trigger set_projects_updated_at
before update on public.projects
for each row execute function public.set_updated_at();

create table if not exists public.billing_customers (
  user_id uuid primary key references auth.users(id) on delete cascade,
  stripe_customer_id text unique,
  plan text not null default 'free',
  status text not null default 'inactive',
  current_period_end timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.billing_customers enable row level security;

drop policy if exists "Users can read own billing customer" on public.billing_customers;
create policy "Users can read own billing customer"
on public.billing_customers
for select
to authenticated
using ((select auth.uid()) = user_id);

drop trigger if exists set_billing_customers_updated_at on public.billing_customers;
create trigger set_billing_customers_updated_at
before update on public.billing_customers
for each row execute function public.set_updated_at();
