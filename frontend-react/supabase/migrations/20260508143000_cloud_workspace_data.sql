create extension if not exists pgcrypto;

alter table public.profiles
  add column if not exists plan text not null default 'free',
  add column if not exists onboarding_completed boolean not null default false,
  add column if not exists last_seen_at timestamptz;

alter table public.projects
  add column if not exists description text,
  add column if not exists source_prompt text,
  add column if not exists preview_html text,
  add column if not exists status text not null default 'active',
  add column if not exists metadata jsonb not null default '{}'::jsonb,
  add column if not exists last_opened_at timestamptz not null default now(),
  add column if not exists archived_at timestamptz;

create index if not exists projects_owner_active_last_opened_idx
on public.projects (owner_id, last_opened_at desc)
where archived_at is null;

create index if not exists projects_owner_status_idx
on public.projects (owner_id, status);

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
